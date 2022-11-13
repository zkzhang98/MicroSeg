"""Modified from SSUL"""
"""
SSUL
Copyright (c) 2021-present NAVER Corp.
MIT License
"""
from tqdm import tqdm
import network
import utils
import os
import time
import random
import argparse
import numpy as np
# import cv2

from torch.utils import data
from datasets import VOCSegmentation, ADESegmentation
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import AverageMeter
from utils.tasks import get_tasks
from utils.memory import memory_sampling_balanced
from utils.loss import UnseenAugLoss
from PIL import Image

torch.backends.cudnn.benchmark = True


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='/data/DB/VOC2012',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc', choices=['voc', 'ade'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None, help="num classes (default: None)")

    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3_resnet101',
                        choices=['deeplabv3_resnet50', 'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--amp", action='store_true', default=False)
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--train_epoch", type=int, default=50,
                        help="epoch number")
    parser.add_argument("--curr_itrs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='warm_poly', choices=['poly', 'step', 'warm_poly'],
                        help="learning rate scheduler")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=32,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")

    parser.add_argument("--loss_type", type=str, default='bce_loss',
                        choices=['ce_loss', 'focal_loss', 'bce_loss'], help="loss type")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")

    parser.add_argument("--pseudo", action='store_true', help="enable pseudo-labeling")
    parser.add_argument("--pseudo_thresh", type=float, default=0.7, help="confidence threshold for pseudo-labeling")
    parser.add_argument("--task", type=str, default='15-1', help="cil task")
    parser.add_argument("--name", type=str, default='default_exp', help="name of exp")
    parser.add_argument("--curr_step", type=int, default=0)
    parser.add_argument("--overlap", action='store_true', help="overlap setup (True), disjoint setup (False)")
    parser.add_argument("--mem_size", type=int, default=0, help="size of examplar memory")
    parser.add_argument("--freeze", action='store_true', help="enable network freezing")
    parser.add_argument("--bn_freeze", action='store_true', help="enable batchnorm freezing")
    parser.add_argument("--unknown", action='store_true', help="enable unknown modeling")
    parser.add_argument("--step", type=str, default=None, help="the step(s) to train/test")

    parser.add_argument("--unseen_multi", action='store_true', help='use multi-unseen class')
    parser.add_argument("--unseen_cluster", type=int, default=5, help='number of unseen cluster')
    parser.add_argument("--unseen_loss", action='store_true', help='use multi-unseen loss')
    parser.add_argument("--loss_tred", action="store_true", help="use tredictional branch")
    parser.add_argument("--not_loss_proposal", action="store_true", help='not to use proposal branch')
    parser.add_argument("--proposal_channel", type=int, default=100)

    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """

    train_transform = et.ExtCompose([
        # et.ExtResize(size=opts.crop_size),
        et.ExtRandomScale((0.5, 2.0)),
        et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    if opts.crop_val:
        val_transform = et.ExtCompose([
            et.ExtResize(opts.crop_size),
            et.ExtCenterCrop(opts.crop_size),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    else:
        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

    if opts.dataset == 'voc':
        dataset = VOCSegmentation
    elif opts.dataset == 'ade':
        dataset = ADESegmentation
    else:
        raise NotImplementedError

    dataset_dict = {}
    dataset_dict['train'] = dataset(opts=opts, image_set='train', transform=train_transform, cil_step=opts.curr_step)

    dataset_dict['val'] = dataset(opts=opts, image_set='val', transform=val_transform, cil_step=opts.curr_step)

    dataset_dict['test'] = dataset(opts=opts, image_set='test', transform=val_transform, cil_step=opts.curr_step)
    # use val to test(for the middle steps of training)

    if opts.curr_step > 0 and opts.mem_size > 0:
        dataset_dict['memory'] = dataset(opts=opts, image_set='memory', transform=train_transform,
                                         cil_step=opts.curr_step, mem_size=opts.mem_size)

    return dataset_dict


def validate(opts, model, loader, device, metrics, alpha=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []

    with torch.no_grad():
        for i, (images, labels, proposals, images_name) in enumerate(loader):

            images = images.to(device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)

            proposals = proposals.to(device, dtype=torch.long, non_blocking=True)
            n_cl = torch.tensor(opts.proposal_channel).to(proposals.device)
            proposals_n = torch.where(proposals != 255, proposals, n_cl)

            proposals_1hot = F.one_hot(proposals_n, opts.proposal_channel+1).permute(0, 3, 1, 2)
            proposals_1hot = proposals_1hot[:, :-1].float()

            outputs, feature_pixel, outputs_pixel = model(images, proposals_1hot)

            if opts.loss_tred and opts.not_loss_proposal:
                outputs = outputs_pixel

            if opts.unseen_multi:
                unknown_clust = outputs[:, :opts.unseen_cluster]
                outputs = torch.cat([torch.sum(unknown_clust, dim=1, keepdim=True),
                                     outputs[:, opts.unseen_cluster:, ]], dim=1)

            if opts.loss_type == 'bce_loss':
                outputs = torch.sigmoid(outputs)
            else:
                outputs = torch.softmax(outputs, dim=1)
            # remove unknown label
            if opts.unknown:
                outputs[:, 1] += outputs[:, 0]
                outputs = outputs[:, 1:]

            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            metrics.update(targets, preds)

        score = metrics.get_results()
    return score


def main(opts):
    assert opts.loss_tred or not opts.not_loss_proposal, 'must use a loss'

    bn_freeze = opts.bn_freeze if opts.curr_step > 0 else False

    target_cls = get_tasks(opts.dataset, opts.task, opts.curr_step)
    opts.num_classes = [len(get_tasks(opts.dataset, opts.task, step)) for step in range(opts.curr_step + 1)]
    len_tasks = len(get_tasks(opts.dataset, opts.task, step=None))
    loss_tred_alpha = None
    if opts.loss_tred:
        loss_tred_alpha = (opts.curr_step + 1) / (len_tasks + 1)
    # opts.num_classes = length of the class in each step(besides curr_step) e.g. [16,5]
    opts.sum_classes = sum(opts.num_classes) - 1
    num_classes = opts.num_classes
    if opts.unknown and not opts.unseen_multi:
        opts.unseen_cluster = 1
    if opts.unknown:  # re-labeling: [unseen, dummy, ...]
        opts.num_classes = [opts.unseen_cluster, 1, num_classes[0] - 1] + num_classes[1:]  # e.g. [5,1,15,5]

    fg_idx = 1 if opts.unknown else 0

    curr_idx = [
        sum(len(get_tasks(opts.dataset, opts.task, step)) for step in range(opts.curr_step)),
        sum(len(get_tasks(opts.dataset, opts.task, step)) for step in range(opts.curr_step + 1))
    ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("==============================================")
    print(f"  task : {opts.task}")
    print(f"  step : {opts.curr_step}")
    print("  Device: %s" % device)
    print("  opts : ")
    print(opts)
    print("==============================================")

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)

    # Set up model
    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }

    model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride, bn_freeze=bn_freeze)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    if opts.curr_step > 0:
        """ load previous model """
        model_prev = model_map[opts.model](num_classes=opts.num_classes[:-1], output_stride=opts.output_stride,
                                           bn_freeze=bn_freeze)
        if opts.separable_conv and 'plus' in opts.model:
            network.convert_to_separable_conv(model_prev.classifier)
        utils.set_bn_momentum(model_prev.backbone, momentum=0.01)
    else:
        model_prev = None

    # Set up metrics
    metrics = StreamSegMetrics(sum(opts.num_classes) - opts.unseen_cluster if opts.unknown else sum(opts.num_classes),
                               dataset=opts.dataset)

    curr_head_num = len(model.classifier.head) - 1
    # Set up optimizer & parameters
    if opts.freeze and opts.curr_step > 0:
        for param in model_prev.parameters():
            param.requires_grad = False

        for param in model.parameters():
            param.requires_grad = False

        for param in model.classifier.head[-1].parameters():  # classifier for new class
            param.requires_grad = True
        for param in model.classifier.head2[-1].parameters():
            param.requires_grad = True
        for param in model.classifier.proposal_head[-1].parameters():
            param.requires_grad = True
        training_params = [{'params': model.classifier.head[-1].parameters(), 'lr': opts.lr},
                           {'params': model.classifier.head2[-1].parameters(), 'lr': opts.lr}]
        training_params.append({'params': model.classifier.proposal_head[-1].parameters(), 'lr': opts.lr})

        if opts.unknown:
            for param in model.classifier.head[0].parameters():  # unknown
                param.requires_grad = True
            training_params.append({'params': model.classifier.head[0].parameters(), 'lr': opts.lr})

            for param in model.classifier.head2[0].parameters():  # unknown
                param.requires_grad = True
            training_params.append({'params': model.classifier.head2[0].parameters(), 'lr': opts.lr})

            for param in model.classifier.head[1].parameters():  # background
                param.requires_grad = True
            training_params.append({'params': model.classifier.head[1].parameters(), 'lr': opts.lr * 1e-4})

            for param in model.classifier.head2[1].parameters():  # background
                param.requires_grad = True
            training_params.append({'params': model.classifier.head2[1].parameters(), 'lr': opts.lr * 1e-4})

            for param in model.classifier.proposal_head[0].parameters():
                param.requires_grad = True
            training_params.append({'params': model.classifier.proposal_head[0].parameters(), 'lr': opts.lr})

            for param in model.classifier.proposal_head[1].parameters():
                param.requires_grad = True
            training_params.append({'params': model.classifier.proposal_head[1].parameters(), 'lr': opts.lr})

    else:
        # for param in model.backbone.parameters():
        #     param.requires_grad = False
        if opts.curr_step > 0:
            for param in model_prev.parameters():
                param.requires_grad = False
            training_params = [{'params': model.backbone.parameters(), 'lr': opts.lr * 1e-2},
                               {'params': model.classifier.aspp.parameters(), 'lr': opts.lr * 1e-3},
                               {'params': model.classifier.head.parameters(), 'lr': opts.lr},
                               {'params': model.classifier.head2.parameters(), 'lr': opts.lr},
                               {'params': model.classifier.proposal_head.parameters(), 'lr': opts.lr}
                               ]
        else:
            training_params = [{'params': model.backbone.parameters(), 'lr': opts.lr * 0.1},
                               {'params': model.classifier.parameters(), 'lr': opts.lr}]

    optimizer = torch.optim.SGD(params=training_params,
                                lr=opts.lr,
                                momentum=0.9,
                                weight_decay=opts.weight_decay,
                                nesterov=True)

    print("----------- trainable parameters --------------")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

    print("-----------------------------------------------")

    def save_ckpt(path):
        torch.save({
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_score": best_score,
        }, path)
        print(os.getcwd())
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints')
    utils.mkdir(os.path.join('checkpoints', opts.name))
    # Restore
    best_score = -1
    cur_itrs = 0
    cur_epochs = 0

    if opts.overlap:
        ckpt_str = "checkpoints/%s/%s_%s_%s_step_%d_overlap.pth"
    else:
        ckpt_str = "checkpoints/%s/%s_%s_%s_step_%d_disjoint.pth"

    if opts.curr_step > 0:  # previous step checkpoint
        opts.ckpt = ckpt_str % (opts.name, opts.model, opts.dataset, opts.task, opts.curr_step - 1)

    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint_all = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        print('prev_ckpt', opts.ckpt)
        checkpoint = checkpoint_all["model_state"]
        model_prev.load_state_dict(checkpoint, strict=True)


        model.load_state_dict(checkpoint, strict=False)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
        del checkpoint_all
    else:
        print("[!] Retrain")

    model = nn.DataParallel(model)
    model = model.to(device)
    model.train()

    if opts.curr_step > 0:
        model_prev = nn.DataParallel(model_prev)
        model_prev = model_prev.to(device)
        model_prev.eval()

        if opts.mem_size > 0:
            memory_sampling_balanced(opts, model_prev)

        # Setup dataloader
    if not opts.crop_val:
        opts.val_batch_size = 1

    dataset_dict = get_dataset(opts)
    train_loader = data.DataLoader(
        dataset_dict['train'], batch_size=opts.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = data.DataLoader(
        dataset_dict['val'], batch_size=opts.val_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = data.DataLoader(
        dataset_dict['test'], batch_size=opts.val_batch_size, shuffle=False, num_workers=4, pin_memory=True)


    print("Dataset: %s, Train set: %d, Val set: %d, Test set: %d" %
          (opts.dataset, len(dataset_dict['train']), len(dataset_dict['val']), len(dataset_dict['test'])))

    if opts.curr_step > 0 and opts.mem_size > 0:
        memory_loader = data.DataLoader(
            dataset_dict['memory'], batch_size=opts.batch_size, shuffle=True, num_workers=4, pin_memory=True,
            drop_last=True)

    total_itrs = opts.train_epoch * len(train_loader)
    val_interval = max(100, total_itrs // 100)
    print(f"... train epoch : {opts.train_epoch} , iterations : {total_itrs} , val_interval : {val_interval}")

    # ==========   Train Loop   ==========#
    if opts.test_only:
        best_ckpt = ckpt_str % (opts.name, opts.model, opts.dataset, opts.task, opts.curr_step)
        print('best_ckpt', best_ckpt)

        checkpoint = torch.load(best_ckpt, map_location=torch.device('cpu'))
        model.module.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        test_score = validate(opts=opts, model=model, loader=test_loader,
                              device=device, metrics=metrics, alpha=loss_tred_alpha)

        class_iou = list(test_score['Class IoU'].values())
        class_acc = list(test_score['Class Acc'].values())

        first_cls = len(get_tasks(opts.dataset, opts.task, 0))  # 15-1 task -> first_cls=16
        print(f"...from 0 to {first_cls - 1} : best/test_before_mIoU : %.6f" % (
                    np.sum(class_iou[:first_cls]) / len(class_iou[:first_cls])))
        print(f"...from {first_cls} to {len(class_iou) - 1} best/test_after_mIoU : %.6f" % (
                    np.sum(class_iou[first_cls:]) / len(class_iou[first_cls:])))
        print(f"...from 0 to {first_cls - 1} : best/test_before_acc : %.6f" % np.mean(class_acc[:first_cls]))
        print(
            f"...from {first_cls} to {len(class_iou) - 1} best/test_after_acc : %.6f" % np.mean(class_acc[first_cls:]))
        return

    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)
    elif opts.lr_policy == 'warm_poly':
        warmup_iters = int(total_itrs * 0.1)
        scheduler = utils.WarmupPolyLR(optimizer, total_itrs, warmup_iters=warmup_iters, power=0.9)

    if not (opts.freeze or opts.freeze_cass):
        pass

    # Set up criterion
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'ce_loss':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    elif opts.loss_type == 'bce_loss':
        criterion = utils.BCEWithLogitsLossWithIgnoreIndex(ignore_index=255,
                                                           reduction='mean')

    scaler = torch.cuda.amp.GradScaler(enabled=opts.amp)

    avg_loss = AverageMeter()
    avg_loss_lua = AverageMeter()
    avg_loss_pixel = AverageMeter()
    avg_time = AverageMeter()

    model.train()

    # =====  Train  =====
    while cur_itrs < total_itrs:
        cur_itrs += 1
        optimizer.zero_grad()
        end_time = time.time()

        """ data load """
        try:
            images, labels, proposals, images_name = train_iter.next()
        except:
            train_iter = iter(train_loader)
            images, labels, proposals, images_name = train_iter.next()
            cur_epochs += 1
            avg_loss.reset()
            avg_loss_lua.reset()
            avg_loss_pixel.reset()
            avg_time.reset()

        images = images.to(device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)
        proposals = proposals.to(device, dtype=torch.long, non_blocking=True)

        """ memory """
        if opts.curr_step > 0 and opts.mem_size > 0:
            try:
                m_images, m_labels, m_sal_maps, _ = mem_iter.next()
            except:
                mem_iter = iter(memory_loader)
                m_images, m_labels, m_sal_maps, _ = mem_iter.next()

            m_images = m_images.to(device, dtype=torch.float32, non_blocking=True)
            m_labels = m_labels.to(device, dtype=torch.long, non_blocking=True)
            m_sal_maps = m_sal_maps.to(device, dtype=torch.long, non_blocking=True)

            rand_index = torch.randperm(opts.batch_size)[:opts.batch_size // 2].cuda()
            images[rand_index, ...] = m_images[rand_index, ...]
            labels[rand_index, ...] = m_labels[rand_index, ...]
            proposals[rand_index, ...] = m_sal_maps[rand_index, ...]

        """ forwarding and optimization """
        with torch.cuda.amp.autocast(enabled=opts.amp):
            n_cl = torch.tensor(opts.proposal_channel).to(proposals.device)
            proposals_n = torch.where(proposals != 255, proposals, n_cl)

            proposals_1hot = F.one_hot(proposals_n, opts.proposal_channel+1).permute(0, 3, 1, 2)
            proposals_1hot = proposals_1hot[:, :-1].float()

            outputs, feature_pixel, outputs_pixel = model(images, proposals_1hot)

            if opts.unseen_multi:
                if not opts.not_loss_proposal:
                    unseen_clust = outputs[:, :opts.unseen_cluster]
                    outputs = torch.cat([torch.sum(unseen_clust, dim=1, keepdim=True),
                                         outputs[:, opts.unseen_cluster:, ]], dim=1)
                if opts.loss_tred:
                    unseen_clust_pixel = outputs_pixel[:, :opts.unseen_cluster]
                    outputs_pixel = torch.cat([torch.sum(unseen_clust_pixel, dim=1, keepdim=True),
                                               outputs_pixel[:, opts.unseen_cluster:, ]], dim=1)

            loss = torch.tensor(0.).to(device)
            lua = torch.tensor(0.).to(device)
            lpixel = torch.tensor(0.).to(device)


            if opts.pseudo and opts.curr_step > 0:
                """ pseudo labeling """
                with torch.no_grad():
                    outputs_prev, feature_pixel_prev, outputs_pixel_prev = model_prev(images, proposals_1hot)

                del proposals_1hot
                del proposals_n

                if opts.unseen_multi:
                    unseen_clust_prev = outputs_prev[:, :opts.unseen_cluster]
                    outputs_prev = torch.cat([torch.sum(unseen_clust_prev, dim=1, keepdim=True),
                                              outputs_prev[:, opts.unseen_cluster:, ]], dim=1)

                if opts.loss_type == 'bce_loss':
                    pred_prob = torch.sigmoid(outputs_prev).detach()
                else:
                    pred_prob = torch.softmax(outputs_prev, 1).detach()

                pred_scores, pred_labels = torch.max(pred_prob, dim=1)
                pseudo_labels = torch.where(
                    (labels <= fg_idx) & (pred_labels > fg_idx) & (pred_scores >= opts.pseudo_thresh),
                    pred_labels,
                    labels)

                if opts.loss_tred:
                    lpixel = criterion(outputs_pixel, pseudo_labels)
                if not opts.not_loss_proposal:
                    loss = criterion(outputs, pseudo_labels)
            else:
                if opts.loss_tred:
                    lpixel = criterion(outputs_pixel, labels)
                if not opts.not_loss_proposal:
                    loss = criterion(outputs, labels)

            if opts.unseen_loss:
                ualoss = UnseenAugLoss()
                lua = ualoss(opts, unseen_clust)

            if opts.curr_step == 0:
                loss = loss + lua + lpixel
            else:
                loss = loss + lua

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        avg_loss.update(loss.item())
        avg_time.update(time.time() - end_time)
        avg_loss_lua.update(lua.item())
        avg_loss_pixel.update(lpixel.item())
        end_time = time.time()

        if (cur_itrs) % 10 == 0:
            print("[%s / step %d] Epoch %d, Itrs %d/%d, Loss=%6f, lua=%6f, ltred=%6f ,Time=%.2f , LR=%.8f" %
                  (opts.task, opts.curr_step, cur_epochs, cur_itrs, total_itrs,
                   avg_loss.avg, avg_loss_lua.avg, avg_loss_pixel.avg, avg_time.avg * 1000,
                   optimizer.param_groups[0]['lr']))

        if val_interval > 0 and (cur_itrs) % val_interval == 0:
            print("validation...")
            model.eval()
            val_score = validate(opts=opts, model=model, loader=val_loader,
                                 device=device, metrics=metrics, alpha=loss_tred_alpha)
            print(val_score)
            print(metrics.to_str(val_score))

            model.train()

            class_iou = list(val_score['Class IoU'].values())
            val_score = np.mean(class_iou[curr_idx[0]:curr_idx[1]] + [class_iou[0]])
            curr_score = np.mean(class_iou[curr_idx[0]:curr_idx[1]])
            print("curr_val_score : %.4f" % (curr_score))

            if curr_score > best_score:  # save best model
                print("... save best ckpt : ", curr_score)
                best_score = curr_score
                save_ckpt(ckpt_str % (opts.name, opts.model, opts.dataset, opts.task, opts.curr_step))

    print("... Training Done")

    if opts.curr_step > 0:
        print("... Testing Best Model")
        best_ckpt = ckpt_str % (opts.name, opts.model, opts.dataset, opts.task, opts.curr_step)

        checkpoint = torch.load(best_ckpt, map_location=torch.device('cpu'))
        model.module.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()

        test_score = validate(opts=opts, model=model, loader=test_loader,
                              device=device, metrics=metrics, alpha=loss_tred_alpha)
        print(metrics.to_str(test_score))

        class_iou = list(test_score['Class IoU'].values())
        class_acc = list(test_score['Class Acc'].values())
        first_cls = len(get_tasks(opts.dataset, opts.task, 0))

        print(class_iou[:first_cls])
        print(np.sum(class_iou[:first_cls]))
        print(len(class_iou[:first_cls]))

        print(f"...from 0 to {first_cls - 1} : best/test_before_mIoU : %.6f" % (
                    np.sum(class_iou[:first_cls]) / len(class_iou[:first_cls])))
        print(class_iou[first_cls:])
        print(f"...from {first_cls} to {len(class_iou) - 1} best/test_after_mIoU : %.6f" % (
                    np.sum(class_iou[first_cls:]) / len(class_iou[first_cls:])))
        print(f"...from 0 to {first_cls - 1} : best/test_before_acc : %.6f" % np.mean(class_acc[:first_cls]))
        print(
            f"...from {first_cls} to {len(class_iou) - 1} best/test_after_acc : %.6f" % np.mean(class_acc[first_cls:]))


if __name__ == '__main__':

    opts = get_argparser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id

    if opts.step is not None:
        steps = [eval(x) for x in opts.step.split(',')]
    else:
        steps = [x for x in range(0, len(get_tasks(opts.dataset, opts.task)))]

    for step in steps:
        opts.curr_step = step
        print("STEP:", opts.curr_step)
        main(opts)
