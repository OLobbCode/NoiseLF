## Learning from Pixel-Level Noisy Label : A New Perspective for Light Field Saliency Detection

### This is a PyTorch implementation of our paper

## Overall

![avatar](https://github.com/OLobbCode/NoiseLF/blob/code/overall.png)



## Prerequisites

- Python 3.6.12

- Pytorch 1.2.0+

- torchvision  0.4.0+

  

## Update

1. We released our code for joint training with depth and appearance, which is also our best performance model.

   

## Usage

### 1. Clone the repository
```shell
git clone https://github.com/OLobbCode/NoiseLF.git
cd NoiseLF-code/
```
### 2. Download the datasets
Download the following datasets and unzip them.
* [DUT-LF](https://pan.baidu.com/share/init?surl=hq135pTjbwuda0VMocOsxw) dataset，fetch code is ‘vecy’. 
* [HFUT](https://github.com/pencilzhang/HFUT-Lytro-dataset) dataset. 
* [LFSD](https://www.eecis.udel.edu/~nianyi/LFSD.htm) dataset. 
* The .txt file link for testing and training is [here](https://pan.baidu.com/s/1uoVtqM8V19fT6rvqgW__cg), code is 'joaa'.
### 3. Train
1. Set the `c.DATA.TRAIN.ROOT` and `c.DATA.TRAIN.LIST` path in `config.py` correctly.
2. We demo using VGG-19 as network backbone and train with a initial lr of 1e-5 for 30 epoches.
3. After training the result model will be stored under `snapshot/exp_noiself` folder.

Note：only support `c.SOLVER.BATCH_SIZE=1`
### 4. Test
For single dataset testing:  you should set  `c.PHASE='test'` in config.py, and set  `c.DATA.TEST.ROOT` ,  `c.DATA.TEST.LIST` as yours.  
```shell
python demo.py 
```
For evaluate :
```shell
python evaluate.py
```
All results saliency maps will be stored under `'Test/Out/exp_noiself_30/'` folders in .png formats.

Thanks to [MOLF](https://github.com/jiwei0921/MoLF).

