This is the implementation of MicroSeg ([NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/99b419554537c66bf27e5eb7a74c7de4-Abstract-Conference.html)).

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

You can also get them from [Baidu Net Disk link](https://pan.baidu.com/s/1WzIwB0ZuJOLvPFnbzRjldA?pwd=dbue).
## Startup

We provide `script_train.py` to help to use our method. For example, you can train MicroSeg-M in default config with the following command:
```
 python script_train.py 15-1 0,1,2,3,4,5 0  --loss_tred  --mem_size 100 --unseen_multi --unseen_loss --unseen_cluster 5   --name  MicroSeg
```
If you want to evaluate model after training , add `--test_only`.

## Acknowledgement
Our code is based on [SSUL](https://github.com/clovaai/SSUL).

## Cite
If you find our work to be helpful, please consider citing us:

[Bibtex](https://proceedings.neurips.cc/paper_files/paper/16674-/bibtex)

```
@inproceedings{NEURIPS2022_99b41955,
 author = {Zhang, Zekang and Gao, Guangyu and Fang, Zhiyuan and Jiao, Jianbo and Wei, Yunchao},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {S. Koyejo and S. Mohamed and A. Agarwal and D. Belgrave and K. Cho and A. Oh},
 pages = {24340--24353},
 publisher = {Curran Associates, Inc.},
 title = {Mining Unseen Classes via Regional Objectness: A Simple Baseline for Incremental Segmentation},
 url = {https://proceedings.neurips.cc/paper_files/paper/2022/file/99b419554537c66bf27e5eb7a74c7de4-Paper-Conference.pdf},
 volume = {35},
 year = {2022}
}
```
