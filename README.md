# GP-DNN
Hierarchical Gaussian process and neural network regression
Code for training multivariate GP-DNNs. See paper by Pradhan et al. (2023) for methodology and application.

The package contains the following:
- *gp_dnn_hyper_tuning.py*: Python file to perform hyper-parameter tuning of GP-DNN models. User needs to specify the following inside the file:
  -  `head_dir`: Local directory to save hyper-parmaeter trained model outputs
  -  `data_dir`: Local data directory
  -  `training_file_name`: Name of pickle file containing training data. For data structure, see file.
  -  `num_samp`: No. of experiments for hyper-parameter tuning
  -  `num_epochs`:  No. of epoch for DNN training
  -  `num_coef`: No of variables for multivariate GP. Note name `coef` in code is used to denote variables/coefficients while name `var` denotes variance.
  -  Need to specify max and min ranges of several hyper-parameters
  Given these specification, the file calls `gp_dnn_hyper_tuning(...)` defined inside *gp_dnn_hyper_tuning.py* for training the GP-DNN 
