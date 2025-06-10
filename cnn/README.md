# Hyperparameter Tuning and Training Pipeline

This directory contains scripts for hyperparameter tuning and training the Convolutional Neural Network (CNN) for the infrasound waveform. The hyperparameter tuning was implemented using optuna, which utilizes a Tree-structured Parzen Estimator (TPE) for selecting hyperparameters. At its core, TPE is probabilistic; therefore, the same optimal hyperparameter is not guaranteed to be found every time. However, the optimal hyperparameter will likely converge at the same solution 

For reproducbility, follow the steps below:

## Files in This Directory

- `tuning_engine_cnn.py`: A script for performing hyperparameter search using Optuna.
- `train_cnn.py`: A script for training a CNN using the hyperparameters found.
---

## Steps to Use

### Step I: Perform Hyperparameter Search
Run the `tuning_engine.py` script to search for optimal hyperparameters, for each of the model configuration. The model configuration of choice can be controlled by changing which feature and label files for each respective run. 

The output has been configured to log to the stream, however a file handler can be configured in the util/helpers.py, if you want to save to file. A file handler was not natively configure since slurm already handles all our logging.

#### Command:
```bash
python tuning_engine.py
```
It is recommended that this be done separately using a high performance computing cluster.

### Step II: Build the CNN Architecture
After the hyperparameter is found, use the optimal hyparameters to configure the CNN Architecture. The different model configuration are seperated by classes. Alternatively you can use the optimal hyperparameter we found during our own search.

Ensure everything runs correctly by running the following, you should see the model summary on your terminal

#### Command:
```bash
python cnn.py
```

### Step III: Train CNN Model
After the CNN architecture is written up, it is time to train the model. . All model can be trained at the same instance, and should take not more than a few hours. Run the following command to train the model.

#### Command:
```bash
python train_cnn.py
```

END OF FILE