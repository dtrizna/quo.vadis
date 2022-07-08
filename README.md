# Quo Vadis

:warning: The model is a research prototype, provided as-is, without warranty of any kind, in pre-alpha state.

## Architecture

Hybrid, modular structure for **malware classification**. Supported modules:

- 1D convolution neural network analysis of filepath at the moment of execution
- 1D convolution neural network analysis of *API call sequence* obtained from [Speakeasy emulator](https://github.com/mandiant/speakeasy/)
- '*Ember*' Gradient Boosted Decision Tree (GBDT) model  (https://arxiv.org/abs/1804.04637)
- '*MalConv*' byte-level convolutional neural network (https://arxiv.org/abs/1710.09435)

Scheme:

<p align="center"><img src="img/composite_scheme.png" width=800><br>

## Dataset and related code

- PE emulation  dataset available in [emulation.dataset.7z](data/emulation.dataset/emulation.dataset.7z)
- Filepath dataset (open sources only, in-the-wild paths used for pre-training are excluded):
  - augmented [samples](data/path.dataset/dataset_malicious_augumented.txt) and [logic](data/path.dataset/augment/augmentation.ipynb)
  - [paths](data/path.dataset/dataset_benign_win10.txt) from clean Windows 10 host


## Usage
Main API interface under: `./models.py`. 
Example usage can be found under `model_api_example.py`:
```
# python model_api_example.py --example

[*] Loading model...
WARNING:root:[!] Loading pretrained weights for ember model from: ./modules/sota/ember/parameters/ember_model.txt
WARNING:root:[!] Loading pretrained weights for filepath model from: ./modules/filepath/pretrained/torch.model
WARNING:root:[!] Using speakeasy emulator config from: ./data/emulation.dataset/sample_emulation/speakeasy_config.json
WARNING:root:[!] Loading pretrained weights for emulation model from: ./modules/emulation/pretrained/torch.model
WARNING:root:[!] Loading pretrained weights for late fusion MultiLayerPerceptron model from: ./modules/late_fustion_model/MultiLayerPerceptron15_ember_emulation_filepaths.model

[*] Legitimate 'calc.exe' analysis...
WARNING:root:[!] Taking current filepath for: evaluation/adversarial/samples_goodware/calc.exe
WARNING:root: [+] 0/0 Finished emulation evaluation/adversarial/samples_goodware/calc.exe, took: 0.19s, API calls acquired: 6
[!] Given path evaluation/adversarial/samples_goodware/calc.exe, probability (malware): 0.000005
[!] Individual module scores:

       ember  filepaths  emulation
0  0.000015    0.00319   0.062108 

WARNING:root: [+] 0/0 Finished emulation evaluation/adversarial/samples_goodware/calc.exe, took: 0.11s, API calls acquired: 6
[!] Given path C:\users\myuser\AppData\Local\Temp\exploit.exe, probability (malware): 0.549334
[!] Individual module scores:

       ember  filepaths  emulation
0  0.000015   0.999984   0.062108 

[*] BoratRAT analysis...
WARNING:root: [+] 0/0 Finished emulation ./b47c77d237243747a51dd02d836444ba067cf6cc4b8b3344e5cf791f5f41d20e, took: 0.25s, API calls acquired: 194

[!] Given path %USERPROFILE%\Downloads\BoratRat.exe, probability (malware): 0.9997
[!] Individual module scores:

       ember  filepaths  emulation
0  0.035511   0.999602    0.96526 

WARNING:root: [+] 0/0 Finished emulation ./b47c77d237243747a51dd02d836444ba067cf6cc4b8b3344e5cf791f5f41d20e, took: 0.25s, API calls acquired: 194

[!] Given path C:\windows\system32\calc.exe, probability (malware): 0.0392
[!] Individual module scores:

       ember  filepaths  emulation
0  0.035511   0.086567    0.96526 
```

## Evaluation

More detailed information about modules and individual tests:

- `./modules/emulation/` - 1D convolutonal pipeline based on API call sequences collected with Windows kernel emulator (we use [Mandiant's Speakeasy](https://github.com/mandiant/speakeasy))
- `./modules/filepaths/` - 1D convolution pipeline for file path classification
- `./modules/sota/` - static PE classification utilizing state-of-the-art ML-models: [MalConv](modules/sota/malconv) or [Ember](modules/sota/ember). Parameters for `sota` models can be downloaded from [here](https://github.com/endgameinc/malware_evasion_competition/tree/master/models).

Performance of this model on the proprietary dataset - 90k PE samples with filepaths from real-world systems:

<center><img src="img/composite_validation_confusionmatrix.png" width=350></center><br>

DET and ROC curves:

<center><img src="img/det_roc_curves.png" width=800></center><br>

Detection rate with fixed False Positive rate:

<center><img src="img/detection_rate_heatmap.png" width=800></center><br>

## Future work

- experiments with **retrained** MalConv / Ember weights on your dataset - it makes sense to evaluate them on the same distribution
  - NOTE: this, however, does not matter since our goal is **not** to compare our modules with MalConv / Ember directly but to improve them. For this reason, it is even better to have original parameters. The main takeaway - adding multiple modules together allows boosting results drastically. At the same time, each of them is noticeably weaker (even the API call module, which is trained on the same distribution).
- try to run GAMMA against composite solution (not just ember/malconv modules) - it looks like attacks are highly targeted. Interesting if it will be able to generate evasive samples against a complete pipeline .. (however, defining that in `secml_malware` might be painful ...)
- work on `CompositeClassifier()` API interface:
  - make it easy to take a PE sample(s) & additional document options (providing PE directory, predefined emulation report directory, etc.)
  - `.update()` to overtrain network with own examples that were previously flagged incorrectly
  - work without submitted `filepath` (only PE mode) - provide paths as separate argument to `.fit()`?
- Additional modules:
  - (a) Autoruns checks (see Sysinternals book for a full list of registries analyzed)
  - (b) network connection information
  - etc.
