# D²M²-BTS: Dual Disentangled Multi-Modal Brain Tumor Segmentation with Private-Shared Feature Enhancement and Uncertainty Estimation
## Usage
### Data Preparation
Please download BraTS 2020 data according to https://www.med.upenn.edu/cbica/brats2020/data.html.
### Training
#### Training on the entire BraTS training set
```bash
python3 train.py --model bts --mixed --trainset
```
#### Breakpoint continuation for training
```bash
python3 train.py --model bts --mixed --trainset --cp checkpoint
```
### Inference
```bash
python3 test.py --model bts --labels --pp --mc_drop --cp checkpoint 
```
