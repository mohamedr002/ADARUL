# Adversarial-Transfer-Learning-for-RUL

This repo contains code of the paper **Adversarial-Transfer-Learning-for-RUL**. It includes code for estimating remaining useful life machine under the assumption that training and testing are from different working environment (distirubtion). 

### Dependencies
This code requires the following:
* python 2.\* or python 3.\*
* Pytorch v1.2+
* TensorboardX
* Plotly

### Data
The model performance is tested on NASA turbofan engines dataset [https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan]. 

### Data Preprocessing
After downloading you can do the data preprocessing steps you can check this file `data_processing.py`

### Usage
- To run the code, we have tried the model using two optimizers, SGD with momentum `Final_SGD.py` and Adam optimizer `Final_Adam.py. The files will show the training results and then print the performance on test set.
- You can also visualize the tensorboard results, using the notebook of `Results Visualization.ipynb`
- Our reports results are based on 5 consecutive runs with calculating the mean and STD. 

