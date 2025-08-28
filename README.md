# Maximum Likelihood Distillation for Robust Modulation Classification

This is the source code to reproduce the experiments of the paper "[Maximum Likelihood Distillation for Robust Modulation Classification](https://ieeexplore.ieee.org/abstract/document/8462493)" by Javier Maroto, Gérôme Bovet and Pascal Frossard.


# Package installation
conda install python
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip install wandb
conda install scikit-learn
conda install tqdm
conda install h5py
conda install pandas
pip install torchattacks
conda install matplotlib

## Reference
If you find this code useful, please cite the following paper:
```bibtex
@INPROCEEDINGS{10096156,
  author={Maroto, Javier and Bovet, Gérôme and Frossard, Pascal},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Maximum Likelihood Distillation for Robust Modulation Classification}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  keywords={Training;Knowledge engineering;Perturbation methods;Neural networks;Modulation;Training data;Rician channels;Neural Networks;Robustness;Maximum Likelihood;Knowledge Distillation;Automatic Modulation Classification},
  doi={10.1109/ICASSP49357.2023.10096156}}
```
