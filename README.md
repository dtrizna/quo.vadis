# Quo Vadis

`data/` - datasets and related code
- [PE emulation dataset](data/emulation.dataset/emulation.dataset.7z)
- Filepath dataset (from open sources only because of Privacy Policy: 
  - augmented [samples](data/path.dataset/dataset_malicious_augumented.txt) and [logic](data/path.dataset/augment/augmentation.ipynb)
  - [paths](data/path.dataset/dataset_benign_win10.txt) from clean Windows 10 host


`modules/filepaths/` - original filepath prediction pipeline based on 1D-convolutional neural network  

<p align="center"><img src="img/potential_scheme.png" width=600><br>

<!--Performance of final model: <center><img src="img/confusion_matrix_on_validation_set_quo.vadis.primus.png" width=350></center><br>-->

`modules/emulation/` -  pipeline based on Windows Kernel emulation based on Speakeasy [emulator](https://github.com/mandiant/speakeasy) from Mandiant


`modules/sota/` - static PE classification state-of-the-art ML-models: [MalConv](modules/sota/malconv) or [Ember](modules/sota/ember)

