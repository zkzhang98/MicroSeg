import sys
import os
import random

assert len(sys.argv) >= 4, f"usage: python script_train.py task step gpus <other commands(including --name)>\n" \
                           f"example python script_train.py 15-5 0,1 0,1 --name MicroSeg"
DATA_ROOT='path/to/your/dataset'
DATASET="voc"
TASK=sys.argv[1]
EPOCH=50
BATCH=16
LOSS="bce_loss"
LR=0.01
CROP_SIZE = 513
MEMORY=100
STEPS=sys.argv[2]
GPUS=sys.argv[3]
other_command = ""
LOG_PATH='checkpoints/logs'
for i in range(4,len(sys.argv)):
    other_command += sys.argv[i]
    other_command += ' '
NAME = ''
for i in range(len(sys.argv)):
    if sys.argv[i] == '--name':
        NAME=sys.argv[i+1]
        break

print(os.getcwd())
print('dataset root:', DATA_ROOT)
if not os.path.exists('checkpoints'):
    os.mkdir('checkpoints')
if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)


os.system(f"CUDA_VISIBLE_DEVICES={GPUS} python main.py --data_root {DATA_ROOT} --model deeplabv3_resnet101 \
          --gpu_id {GPUS} --step {STEPS} --crop_val --lr {LR} --crop_size {CROP_SIZE}\
    --batch_size {BATCH} --train_epoch {EPOCH}  --loss_type {LOSS} \
    --dataset {DATASET} --task {TASK} --overlap --lr_policy poly \
    --pseudo --mem_size {MEMORY} "
    "--freeze  --bn_freeze --unknown --amp "
    f"{other_command} "
    )




