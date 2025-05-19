A baseline Prompt-State-Action Causal Modeling for Maze World

## Install


## Training
Reconfigure the settings in `config.yaml`, start training with the following command.
```bash
export CUDA_VISIBLE_DEVICES=...
python train.py config.yaml --configs key1=value1 key2=value2 ...
```

## Validation
```bash
export CUDA_VISIBLE_DEVICES=...
python validate.py config.yaml --configs key1=value1 key2=value2 ...
```
