############
#           Perform hyper-parameter tuning of GP-DNN
############
import numpy as np
import os
import scipy.stats as st
import tensorflow as tf
import pickle

from utils_gp_dnn import *
from gp_dnn_training import *

tf.keras.backend.set_floatx("float64")

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(101)

## Note name "coef" is used to denote variables of the multivariate GP
## Name "var" in below code denotes variance and not variable
############# Data loading
# training_data = {
#     'x_train': <<n_train*n_coef \times n_features array>>
#     'y_train': <<n_train*n_coef \times 2 array>>
#                 2nd column indicates the index of coeffient/variable,
#                 this is needed to determine which coeffient the
#                 training example is for. Index \in [0, num_coef-1].
#     'x_val': <<n_val*n_coef \times n_features array>>
#     'y_val': <<n_val*n_coef \times 2 array>>
#     'x_test': <<n_test*n_coef \times n_features array>>
#     'y_test': <<n_test*n_coef \times 2 array>>
# }
head_dir = "./"  # <Enter path to directory to save trained models here>
data_dir = "./Data/"  # <Enter path to data directory >
training_file_name = "training_data.pickle"  # <Enter training data pickle file here>
with open(data_dir + training_file_name, "rb") as file_pi:
    training_data = pickle.load(file_pi)
###############################


####################
num_samp = 5000  # No. of experiments for hyper-parameter tuning
num_epochs = 100  # No. of epoch for DNN training
num_coef = 4  # No of variables for multivariate GP.
######################

########################### Sample hyper-parameters


num_layers_samp = np.random.randint(1, 4, size=num_samp)  # No. of DNN layers
num_nodes_samp = np.random.randint(30, 130, size=num_samp)  # No. of nodes in each layer
num_outputs_samp = np.random.randint(1, 30, size=num_samp)  # No. of outputs

use_dropout_samp = np.random.randint(2, size=num_samp).astype(
    bool
)  # Boolean flag for using dropout
droprates_samp = 0.01 + (0.30 - 0.01) * np.random.rand(num_samp)  # Dropout rate

# Boolean flag for using batch normalization
# Not using batchnorm if 1 upper limit in np.random.randint(). Change to 2 if using
use_minibatch_samp = np.random.randint(1, size=num_samp).astype(bool)
minibatch_size_prior = [32, 64, 128, 256, 512, 1024]  # Mini-batch size
minibatch_size_samp = [
    minibatch_size_prior[i]
    for i in np.random.randint(0, high=len(minibatch_size_prior), size=num_samp)
]

# Max/Min noise variance for each variable
noise_vars = [[0.2, 0.5], [0.2, 0.7], [0.4, 0.98], [0.4, 0.98]]
noise_var_samp = [
    [
        noise_vars[i][0] + (noise_vars[i][1] - noise_vars[i][0]) * np.random.rand()
        for i in range(num_coef)
    ]
    for _ in range(num_samp)
]

# Max/Min correlation coefficient b/w each unique pair pf variables specified as
# [[[v1,v2],[v1,v3],[v1,v4]],[[v2,v3],[v2,v4]], [v3,v4]]
corr_coefs = [
    [[-0.7, -0.25], [0.01, 0.2], [-0.4, -0.05]],
    [[-0.15, 0.15], [-0.05, 0.15]],
    [[-0.9, -0.4]],
]
corr_coef_samp = [
    [
        [cc2[0] + (cc2[1] - cc2[0]) * np.random.rand() for cc2 in cc1]
        for cc1 in corr_coefs
    ]
    for _ in range(num_samp)
]

l2_reg_samp = st.loguniform.rvs(
    0.001, 10, size=num_samp
)  # l-2 regularization coefficient
learning_rate_samp = st.loguniform.rvs(
    0.001, 0.5, size=num_samp
)  # Learning rate for Adam optimizer
random_seed_samp = np.random.randint(1e3, 1e5, size=num_samp)  # Random seed
############################


############################### DL training
for j in range(num_samp):
    hyper_params = {
        "random_seed": random_seed_samp[j],
        "num_layers": num_layers_samp[j],
        "num_nodes": num_nodes_samp[j],
        "num_outputs": num_outputs_samp[j],
        "num_coef": num_coef,
        "corr_coef_all_coef_pairs": corr_coef_samp[j],
        "noise_var_all_coef": noise_var_samp[j],
        "learning_rate": learning_rate_samp[j],
        "l2_reg_coef": l2_reg_samp[j],
        "use_minibatch": use_minibatch_samp[j],
        "minibatch_size": minibatch_size_samp[j],
        "use_dropout": use_dropout_samp[j],
        "droprates": droprates_samp[j],
    }
    model_output_path = head_dir + "model_" + str(j) + "/"
    gp_dnn_training(training_data, num_epochs, model_output_path, hyper_params)
