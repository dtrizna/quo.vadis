# Downloaded weights from SOREL-20 paper

This works, however, produce accuracies close to untrained network:

```bash
$ python3 pretrain.py --epochs 100 --optimizer adamw -lr 1e-3 --model-input sorel FFNN/epoch_10.pt
[*] Reading dataset... Took: 242.18s
WARNING:root: [*] Mon Jul 18 12:45:37 2022: Loading PyTorch model state from sorelFFNN/epoch_10.pt
WARNING:root: [*] Mon Jul 18 12:45:37 2022: Started epoch: 1
WARNING:root: [*] Mon Jul 18 12:45:38 2022: Train Epoch: 1 [0/672411 (0%)]      Loss: 2.253491  Acc: 55.00 | Elapsed: 1.13s
WARNING:root: [*] Mon Jul 18 12:45:41 2022: Train Epoch: 1 [102400/672411 (15%)]        Loss: 0.660506  Acc: 52.28 | Elapsed: 2.83s
WARNING:root: [*] Mon Jul 18 12:45:44 2022: Train Epoch: 1 [204800/672411 (30%)]        Loss: 0.677007  Acc: 52.36 | Elapsed: 2.77s
...
```

Discarded...
