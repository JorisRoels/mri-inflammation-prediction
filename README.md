# Inflammation prediction along the SI joint region in MRI data

This repository corresponds to the paper 
- Joris Roels, Thomas Renson, Ann-Sophie De Craemer, Manouk De Hooge, Philippe Caron, Dirk Elewaut, Yvan Saeys, 
  "Inflammation prediction along the SI joint region in MRI data", https://www.overleaf.com/project/607e8204177bcec68052a6e4
  
## Installation

To get started with this code, you will have to download the code, and install the dependencies: 
```bash
git clone https://github.com/JorisRoels/mri-inflammation-prediction
cd mri-inflammation-prediction
pip install -r requirements.txt
```

Note: the repository was developed and tested in an isolated Anaconda environment with Python 3.8. We recommend to do the same thing. 

## Data

If you would like to reproduce our work, you can download the data and annotations to your directory of choice as follows: 
```bash
DATA_PATH=/path/to/your/data
wget -P $DATA_PATH linkwillfollow.com/data.tar.gz
tar -xf data.tar.gz
```

## Usage

### Pre-processing

After you have installed the repository and downloaded the data, you can pre-process the data as follows:
```bash
python data/preprocess.py --data-dir $DATA_PATH/data --datasets BEGIANT,HEALTHY_CONTROLS,POPAS --merge
```
This effectively pre-processes the three studied datasets from the paper and merges them. 

To visualize the pre-processed data, you can simply run: 
```bash
python data/visualize.py --data-dir $DATA_PATH/data/merged --print-scores
```

### SI joint detection
To detect the sacroilliac (SI) joints, we train an [EfficientDet](https://openaccess.thecvf.com/content_CVPR_2020/html/Tan_EfficientDet_Scalable_and_Efficient_Object_Detection_CVPR_2020_paper.html) model to detect left and right SI joints. 
```bash
python train/efficientdet.py --annotations $DATA_PATH/data/si-joints
```

You can then test the trained model by running: 
```bash
python test/efficientdet.py --data-dir $DATA_PATH/data/merged --model-checkpoint train/logs/efficientdet/final-model.ckpt
```

### Illium & sacrum segmentation
Segmentation of the illium and sacrum is performed by training a [U-Net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) on noisy labels, obtained by the Carving workflow in [ilastik](https://www.ilastik.org/). 
```bash
python train/unet.py --data $DATA_PATH/data/carving/data --labels $DATA_PATH/data/carving/labels_illium
python train/unet.py --data $DATA_PATH/data/carving/data --labels $DATA_PATH/data/carving/labels_sacrum
```

You can then test the trained model by running: 
```bash
python test/unet.py --data-dir $DATA_PATH/data/merged --model-checkpoint-illium train/logs/unet/illium/final-model.ckpt --model-checkpoint-sacrum train/logs/unet/sacrum/final-model.ckpt
```

### Inflammation prediction
The inflammation predictor can be trained by providing it with an EfficientDet SI joint detector and an illium and sacrum segmentation U-Net: 
```bash
python train/classifier.py --data-dir $DATA_PATH/data/merged --si-joint-model train/logs/efficientdet/final-model.ckpt --illium-model train/logs/unet/illium/final-model.ckpt --sacrum-model train/logs/unet/sacrum/final-model.ckpt
```

