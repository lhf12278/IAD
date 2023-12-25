![Python >=3.6](https://img.shields.io/badge/Python->=3.6-yellow.svg)
![PyTorch >=1.7](https://img.shields.io/badge/PyTorch->=1.7-blue.svg)

# Interactive Attack-Defense for Generalized Person Re-Identification [[pdf]](wating)
The *official* repository for [Interactive Attack-Defense for Generalized Person Re-Identification](wating).

## Requirements

### Installation
```bash
pip install -r requirements.txt
```
We recommend to use /Python=3.8 /torch=1.10.1 /torchvision=0.11.2 /timm=0.6.13 /cuda==11.3 /faiss-gpu=1.7.2/ 24G RTX 3090 or RTX 4090 for training and evaluation. If you find some packages are missing, please install them manually. 

### Prepare Datasets

```bash
mkdir data
```

Download the datasets:
- [Market-1501](https://drive.google.com/file/d/1pYM3wruB8TonHLwMQ_g1KAz-UqRrH006/view?usp=drive_link)
- [MSMT17](https://drive.google.com/file/d/1TD3COX3laYIpXNvKN6vazv_7x8PNdYkI/view?usp=drive_link)


Then unzip them and rename them under the directory like

```
data
├── market1501
│   └── bounding_box_train
│   └── bounding_box_test
│   └── ..
├── MSMT17
│   └── train
│   └── test
│   └── ..
```


## Acknowledgment
Our implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.

[LUPerson](https://github.com/DengpanFu/LUPerson), [DINO](https://github.com/facebookresearch/dino), [TransReID](https://github.com/damo-cv/TransReID),
[TransReID-SSL](https://github.com/damo-cv/TransReID-SSL)

## Citation

If you find this code useful for your research, please cite our paper

```
wating
```

## Contact

If you have any question, please feel free to contact us. E-mail: [zclry588@gmail.com](mailto:zclry588@gmail.com)
