![Python 3.6](https://img.shields.io/badge/python-3.6.9-green.svg)
![Packagist](https://img.shields.io/badge/Pytorch-0.4.1-red.svg)
![Last Commit](https://img.shields.io/github/last-commit/Amazingren/CIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-blue.svg)]((https://github.com/Amazingren/CIT/graphs/commit-activity))
![Contributing](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)
![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)

# Cloth Interactive Transformer (CIT)

[Cloth Interactive Transformer for Virtual Try-On](https://arxiv.org/abs/2104.05519) <br> 
[Bin Ren](https://scholar.google.com/citations?user=Md9maLYAAAAJ&hl=en)<sup>1</sup>, [Hao Tang](http://disi.unitn.it/~hao.tang/)<sup>1</sup>, Fanyang Meng<sup>2</sup>, Runwei Ding<sup>3</sup>, [Ling Shao](https://scholar.google.com/citations?user=z84rLjoAAAAJ&hl=en)<sup>4</sup>, [Philip H.S. Torr](https://scholar.google.com/citations?user=kPxa2w0AAAAJ&hl=en)<sup>5</sup>, [Nicu Sebe](https://scholar.google.com/citations?user=stFCYOAAAAAJ&hl=en)<sup>16</sup>. <br> 
<sup>1</sup>University of Trento, Italy, 
<sup>2</sup>Peng Cheng Laboratory, China,
<sup>3</sup>Peking University Shenzhen Graduate School, China, <br>
<sup>4</sup>Inception Institute of AI, UAE,
<sup>5</sup>University of Oxford, UK,
<sup>6</sup>Huawei Research Ireland, Ireland.<br>

The repository offers the official implementation of our paper in PyTorch.
The code and pre-trained models are tested with pytorch 0.4.1, torchvision 0.2.1, opencv-python 4.1, and pillow 5.4 (Python 3.6).

In the meantime, check out our recent paper [XingGAN](https://github.com/Ha0Tang/XingGAN) and [XingVTON](https://github.com/Ha0Tang/XingVTON).

## Usage
This pipeline is a combination of consecutive training and testing of Cloth Interactive Transformer (CIT) Matching block based GMM and CIT Reasoning block based TOM. GMM generates the warped clothes according to the target human. Then, TOM blends the warped clothes outputs from GMM into the target human properties, to generate the final try-on output.

1) Install the requirements
2) Download/Prepare the dataset
3) Train the CIT Matching block based GMM network
4) Get warped clothes for training set with trained GMM network, and copy warped clothes & masks inside `data/train` directory
5) Train the CIT Reasoning block based TOM network
6) Test CIT Matching block based GMM for testing set
7) Get warped clothes for testing set, copy warped clothes & masks inside `data/test` directory
8) Test CIT Reasoning block based TOM testing set

## Installation
This implementation is built and tested in PyTorch 0.4.1.
Pytorch and torchvision are recommended to install with conda: `conda install pytorch=0.4.1 torchvision=0.2.1 -c pytorch`

For all packages, run `pip install -r requirements.txt`

## Data Preparation
For training/testing VITON dataset, our full and processed dataset is available here: https://1drv.ms/u/s!Ai8t8GAHdzVUiQQYX0azYhqIDPP6?e=4cpFTI. After downloading, unzip to your own data directory `./data/`.

## Training
Run `python train.py` with your specific usage options for GMM and TOM stage.

For example, GMM: ```python train.py --name GMM --stage GMM --workers 4 --save_count 5000 --shuffle```.
Then run test.py for GMM network with the training dataset, which will generate the warped clothes and masks in "warp-cloth" and "warp-mask" folders inside the "result/GMM/train/" directory. 
Copy the "warp-cloth" and "warp-mask" folders into your data directory, for example inside "data/train" folder.

Run TOM stage, ```python train.py --name TOM --stage TOM --workers 4 --save_count 5000 --shuffle```

## Evaluation
We adopt four evaluation metrics in our work for evaluating the performance of the proposed XingVTON. There are Jaccard score (JS), structral similarity index measure (SSIM), learned perceptual image patch similarity (LPIPS), and Inception score (IS).

Note that JS is used for the same clothing retry-on cases (with ground truth cases) in the first geometric matching stage, while SSIM and LPIPS are used for the same clothing retry-on cases (with ground truth cases) in the second try-on stage. In addition, IS is used for different clothing try-on (where no ground truth is available).

### For JS 
- Step1: Run```python test.py --name GMM --stage GMM --workers 4 --datamode test --data_list test_pairs_same.txt --checkpoint checkpoints/GMM_pretrained/gmm_final.pth```
then the parsed segmentation area for current upper clothing is used as the reference image, accompanied with generated warped clothing mask then:
- Step2: Run```python metrics/getJS.py```

### For SSIM
After we run test.py for GMM network with the testibng dataset, the warped clothes and masks will be generated in "warp-cloth" and "warp-mask" folders inside the "result/GMM/test/" directory. Copy the "warp-cloth" and "warp-mask" folders into your data directory, for example inside "data/test" folder. Then:
- Step1: Run TOM stage test ```python test.py --name TOM --stage TOM --workers 4 --datamode test --data_list test_pairs_same.txt --checkpoint checkpoints/TOM_pretrained/tom_final.pth```
Then the original target human image is used as the reference image, accompanied with the generated retry-on image then:
- Step2: Run ```python metrics/getSSIM.py```

### For LPIPS
- Step1: You need to creat a new virtual enviriment, then install PyTorch 1.0+ and torchvision;
- Step2: Run ```sh metrics/PerceptualSimilarity/testLPIPS.sh```;

### For IS
- Step1: Run TOM stage test ```python test.py --name TOM --stage TOM --workers 4 --datamode test --data_list test_pairs.txt --checkpoint checkpoints/TOM_pretrained/tom_final.pth```
- Step2: Run ```python metrics/getIS.py```

## Inference
The pre-trained models are provided [here](https://drive.google.com/drive/folders/12SAalfaQ--osAIIEh-qE5TLOP_kJmIP8?usp=sharing). Download the pre-trained models and put them in this project (./checkpoints)
Then just run the same step as Evaluation to test/inference our model.

## Acknowledgements
This source code is inspired by [CP-VTON](https://github.com/sergeywong/cp-vton), [CP-VTON+](https://github.com/minar09/cp-vton-plus). We are extremely grateful for their public implementation.

## Citation
If you use this code for your research, please consider giving a star :star: and citing our [paper](https://arxiv.org/abs/2104.05519) :t-rex::

CIT
```
@article{ren2021cloth,
  title={Cloth Interactive Transformer for Virtual Try-On},
  author={Ren, Bin and Tang, Hao and Meng, Fanyang and Ding, Runwei and Shao, Ling and Torr, Philip HS and Sebe, Nicu},
  journal={arXiv preprint arXiv:2104.05519},
  year={2021}
}
```


## Contributions
If you have any questions/comments/bug reports, feel free to open a github issue or pull a request or e-mail to the author Bin Ren ([bin.ren@unitn.it](bin.ren@unitn.it)).