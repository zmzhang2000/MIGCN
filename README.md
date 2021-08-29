# Multi-Modal Interaction Graph Convolutional Network for Temporal Language Localization in Videos

## Environment Settings
We use the framework PyTorch.
* PyTorch version: 1.4.0
* Python version: 3.7.0

## Code Running
For model training on Charades-STA dataset with I3D feature, run:
```
python main.py
```
For model testing, run:
```
python main.py --test --model_load_path checkpoints/best-model
```
For other dataset, feature and hyper-parameters, run:
```
python main.py -h
```
to get more help information.