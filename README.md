# IVSA-GZSL
Codes for the Multimedia Systems 2024 paper: Indirect Visual-Semantic Alignment for Generalized Zero-Shot Recognition [[paper](https://doi.org/10.1007/s00530-024-01313-z)].
![](./images/framework.png)

## Indirect Visual-Semantic Alignment for Generalized Zero-Shot Recognition
### Dependencies
- Python 3.8.5
- Pytorch 1.7.1
- pyyaml
- pytorch_metric_learning

### Datasets
Please download the [datasets](https://drive.google.com/drive/folders/1sL2wrQmwUtoEvCTaEpVYZsIkNIde2wC2?usp=sharing) 'data' in the directory './IVSA-GZSL/', so the path is
'./IVSA-GZSL/data/'

### Pretrained Models
Please download the [pretrained weight](https://drive.google.com/drive/folders/1sL2wrQmwUtoEvCTaEpVYZsIkNIde2wC2?usp=sharing) 'pretrained' in the directory './IVSA-GZSL/', so the path is './IVSA-GZSL/pretrained/'.

### Evaluate
- If you want to evaluate all datasets, run:  
```
python eval_AllDataset.py
```
- Evaluate specific datasets set by gzsl, for example:  
if you want to evaluate 'CUB' dataset, then modify `dataset: CUB` in '/config/Control.yaml'. And run:
```
python eval.py
```

### Train
- All parameter settings are in '/config/':
  - train_config_gzsl_APY.yaml
  - train_config_gzsl_AWA2.yaml
  - train_config_gzsl_CUB.yaml
  - train_config_gzsl_FLO.yaml
  - train_config_gzsl_SUN.yaml
- If you want to train the CUB dataset, then modify `dataset: CUB` in '/config/Control.yaml'. And run:
```
python train.py
```

### Citation
If you find this useful, please cite:
```
@article{chen2024indirect,
  title={Indirect visual-semantic alignment for generalized zero-shot recognition},
  author={Chen, Yan-He and Yeh, Mei-Chen},
  journal={Multimedia Systems},
  volume={30},
  number={2},
  pages={111},
  year={2024},
  publisher={Springer}
}
```
