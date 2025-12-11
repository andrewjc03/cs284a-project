# cs284a-project

Final project for CS 284A @ UCI

Dataset: APTOS 2019 Blindness Kaggle Competition


## Deep Learning for Diabetic Retinopathy Severity Classification Using Retinal Fundus Images 
#### Authors: Andrew Collins, Kevin Fang, and Zachary Katz
---

## Instructions to run demo on Google Colab

1. Upload ```DemoCode.ipynb``` to Google Drive.
2. Open ```DemoCode.ipynb``` in Google Colab.
3.Execute all cells in ```DemoCode.ipynb```
Google Colab has all of the necessary libraries pre-installed. This repository will be cloned inside Google Colab and the necessary datasets will be installed. The code also is configured to used GPUs, if available.

## Instructions to run demo locally 
1. Clone this repository! Run ```git clone https://github.com/andrewjc03/cs284a-project.git``` then ```cd cs284a-project```
2. Install the necessary libraries (in a virtual environment if you wish) with the following command: ```pip install -r requirements.txt``` (there is a cell in ```DemoCode.ipynb``` to do this for you).
** Note the version of PyTorch that is installed. Google Colab uses 2.9.0 but for older versions of PyTorch, the ```GradScaler``` may cause issues.
3. Execute all cells in DemoCode.ipynb (VS Code or Jupyter Notebook).
```requirements.txt``` was generated using the command ```!pip freeze > requirements.txt``` in Google Colab. Extraneous libraries were removed.

### Troubleshooting Local Run

If ```PyTorch>=2.9.0``` cannot be installed, try an older version of ```PyTorch```. In this case, change the import statement ```from torch.amp import autocast, GradScaler``` to ```from torch.cuda.amp import autocast, GradScaler``` if it raises an issue, and change ```scaler = GradScaler(device="cuda")``` to ```scaler = GradScaler()```. 



---
In our demo, we use a smaller subset of the original data, which we got from the training dataset of [this Kaggle competition](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data). Our demo uses 696 images for training and 184 images for testing.

For our project, we tested the effectiveness of three different backbone models (ResNet50, EfficientNet-B3, Inception-V3) for identifying Diabetic Retinopathy Severity from retinal fundus images. In order to test our code, make sure that this notebook file is in the same directory as:

- The training data subset images folder (train_images_subset/)
- The testing data subset images folder (test_images_subset/)
- training labels subset CSV (train_split.csv)
- testing labels subset CSV (test_split.csv)

Run all the cells in this notebook to load and test all 3 backbone models and display their results. The code is device agnostic, so feel free to use either CPU or GPU, although using CPUs will noticably reduce the completion speed.

