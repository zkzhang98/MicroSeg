This is the implementation of MicroSeg.

## Requirements
All experiments in this paper are done with following environments:

- CUDA 11.6
- python (3.6.13)
- pytorch (1.7.1+cu110)
- torchvision (0.8.2+cu110)
- numpy (1.19.2)
- matplotlib
- pillow

## Dataset preparing

Organize datasets in the following structure.
```
path_to_your_dataset/
    VOC2012/
        Annotations/
        ImageSet/
        JPEGImages/
        SegmentationClassAug/
        proposal100/
        
    ADEChallengeData2016/
        annotations/
            training/
            validation/
        images/
            training/
            validation/
        proposal_adetrain/
        proposal_adeval/
```
You can get [proposal100](https://drive.google.com/file/d/1FxoyVa0I1IEwtW2ykGlNf-JkOYkK80E6/view?usp=sharing), [proposal_adetrain](https://drive.google.com/file/d/1kWfPNhoUnYz0uPuHJUALxiqvVqlCKrwW/view?usp=sharing), [proposal_adeval](https://drive.google.com/file/d/16xNMO4siqJXr5A03ywQDXU0F1Ld5OFtw/view?usp=sharing) here.

Baidu Net Disk link: https://pan.baidu.com/s/1WzIwB0ZuJOLvPFnbzRjldA?pwd=dbue
## Startup

We provide `script_train.py` to help to use our method. For example, you can train MicroSeg-M in default config with the following command:
```
 python script_train.py 15-1 0,1,2,3,4,5 0  --loss_tred  --mem_size 100 --unseen_multi --unseen_loss --unseen_cluster 5   --name  MicroSeg
```
If you want to evaluate model after training , add `--test_only`.

## Acknowledgement
Our code is based on [SSUL](https://github.com/clovaai/SSUL).
