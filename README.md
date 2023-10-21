# GP-DNN
Hierarchical Gaussian process and neural network regression.
Python code for training multivariate GP-DNNs using `tensorflow` and `tensorflow-probability` (see *requirements.txt* file). Refer to paper by Pradhan et al. (2023) for description of methodology and application.

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

- *gp_dnn_training.py*: Contains function `gp_dnn_training` for training the GP-DNN.
  - Requires the training data, # of epochs, model output path and hyper-parameters
  - Saves following Tensorflow outputs in provided output path
    - Model weights saved as model checkpoints, see [TF documentation](https://www.tensorflow.org/guide/checkpoint)
    - Training history in a *history.pickle*
    - Model hyper-parameters and training metrics in a *params.json* file

- *utils_gp_dnn.py*: Contains the following:
  - `GP_DNN_marginal_neglikelihood_loss`: TF subclass for GP-DNN marginal likelihood loss functions
  - `GP_DNN_likelihood`: TF subclass for GP-DNN likelihood metric
  - `GP_DNN_RMSE`: TF subclass for GP-DNN mean RMSE metric
  - `cokriging_covariance`: Function for constructing covariance matrix using Matérn kernels

Users may choose the best model from hyper-parameter tuning based on saved metrics and perform subsequent analyses