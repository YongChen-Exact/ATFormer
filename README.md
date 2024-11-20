# ATFormer
The codes for the work "ATFormer: Advanced transformer for medical image segmentation".

![ATFormer](img/Architecture_overview.jpg?raw=true)

Parts of codes are borrowed from [TransUNet](https://github.com/Beckschen/TransUNet). For detailed configuration of the dataset, please refer to [TransUNet](https://github.com/Beckschen/TransUNet).

## Environment

- Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

## Train/Test

- Run the train script on ACDC dataset. The batch size we used is 24. 

- Train

```bash
python train.py
```

- Test 

```bash
python test.py
```

## Acknowledgements

This repository makes liberal use of code from:

* [TransUnet](https://github.com/Beckschen/TransUNet)
* [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)
* [MISSFormer](https://github.com/ZhifangDeng/MISSFormer)
* [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)

## Citation

@article{chen2023atformer,
  title={ATFormer: Advanced transformer for medical image segmentation},
  author={Chen, Yong and Lu, Xuesong and Xie, Qinlan},
  journal={Biomedical Signal Processing and Control},
  volume={85},
  pages={105079},
  year={2023},
  publisher={Elsevier}
}
