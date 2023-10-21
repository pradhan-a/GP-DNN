import numpy as np
import os
import tensorflow as tf
import shutil
import json
import pickle
import time
import random
from utils_gp_dnn import *


def gp_dnn_training(training_data, epochs, model_output_path, hyper_params):
    # Function for training GP-DNN
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
    # epochs: integer for the number of training epochs
    # model_output_path: string for directory to save model outputs in
    # hyper_params: dict containing hyper-parameters, see below for complete list
    #
    # Saves trained model files in model_output_path
    ##################################

    ############# Read training data
    x_train = training_data["x_train"]
    y_train = training_data["y_train"]

    x_val = training_data["x_val"]
    y_val = training_data["y_val"]

    x_test = training_data["x_test"]
    y_test = training_data["y_test"]

    num_inputs = x_train.shape[-1]
    #############

    ############# Read hyper_paramerters
    random_seed = hyper_params["random_seed"]
    num_layers = hyper_params["num_layers"]
    num_nodes = hyper_params["num_nodes"]
    num_outputs = hyper_params["num_outputs"]
    num_coef = hyper_params["num_coef"]
    corr_coef = np.eye(num_coef) * 0.5
    for k1 in range(num_coef - 1):
        for k2 in range(k1 + 1, num_coef):
            corr_coef[k1, k2] = hyper_params["corr_coef_all_coef_pairs"][k1][
                k2 - 1 - k1
            ]
    corr_coef = corr_coef + np.transpose(corr_coef)
    corr_coef = corr_coef.astype("float64")
    noise_var = hyper_params["noise_var_all_coef"]
    learning_rate = hyper_params["learning_rate"]
    l2_reg = hyper_params["l2_reg_coef"]
    use_minibatch = hyper_params["use_minibatch"]
    minibatch_size = hyper_params["minibatch_size"]
    use_dropout = hyper_params["use_dropout"]
    droprates = hyper_params["droprates"]
    #############

    ############# Initialize variables
    eval_var = [i for i in range(num_coef)]
    random.seed(int(random_seed))
    np.random.seed(int(random_seed) + 1)
    tf.random.set_seed(int(random_seed) + 2)
    keep_training_flag = True
    ##############

    # Make new directory for storing models
    if os.path.isdir(model_output_path):
        shutil.rmtree(model_output_path, ignore_errors=True)
    os.mkdir(model_output_path)
    checkpoint_path = model_output_path + "/model_ckpt.ckpt"

    ################## Build tf functional API for DNN
    inputs = tf.keras.Input(shape=(num_inputs,))

    for k in range(num_layers):
        if k == 0:
            x = tf.keras.layers.Dense(
                num_nodes, kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
            )(inputs)
        else:
            x = tf.keras.layers.Dense(
                num_nodes, kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
            )(x)

        if use_minibatch:
            x = tf.keras.layers.BatchNormalization()(x)
            batch_size = minibatch_size
        else:
            batch_size = x_train.shape[0]

        x = tf.keras.layers.Activation("relu")(x)

        if use_dropout:
            drop_rate = droprates
        else:
            drop_rate = 0.0
        x = tf.keras.layers.Dropout(rate=drop_rate)(x)

    outputs = tf.keras.layers.Dense(
        num_outputs,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
    )(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    #############

    ############### Specify the optimizer, loss function and metrics
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = GP_DNN_marginal_neglikelihood_loss(
        noise_var=noise_var, corr_coef=corr_coef
    )
    cokrigrmse_metric = GP_DNN_RMSE(
        model,
        x_train=x_train,
        y_train_obs=y_train,
        noise_var=noise_var,
        corr_coef=corr_coef,
        eval_var=eval_var,
        name="gp_dnn_RMSE",
    )
    cokrigLL_metric = GP_DNN_likelihood(
        model,
        x_train=x_train,
        y_train_obs=y_train,
        noise_var=noise_var,
        corr_coef=corr_coef,
        eval_var=eval_var,
        name="gp_dnn_LL",
    )

    ############## Prepare the training & validation dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=batch_size).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(batch_size)
    ####################

    param_grid = {
        "num_layers": num_layers,
        "num_nodes": num_nodes,
        "num_outputs": num_outputs,
        "use_dropout": str(use_dropout),
        "drop_rate": drop_rate,
        "use_minibatch": str(use_minibatch),
        "batch_size": batch_size,
        "noise_var": noise_var,
        "corr_coef": corr_coef,
        "learning_rate": learning_rate,
        "l2_reg": l2_reg,
        "checkpoint_path": checkpoint_path,
        "tf_keras_seed": int(random_seed),
    }
    # Store hyper-params ad metrics in a params.json file
    with open(model_output_path + "/params.json", "w") as f:
        json.dump(param_grid, f, cls=NpEncoder, indent=2)

    history_metrics = {"loss_fn_train": [], "metric_LL_val": []}
    best_metric_val = np.inf
    ############ Training
    print("\n Beginning training")
    for epoch in range(epochs):
        start_time = time.time()

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                x_tilde = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, x_tilde)

            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Log every 20 batches.
            if (epoch % 10 == 0) & (step % 2 == 0):
                print("\n Training epoch %d" % (epoch,))
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %d samples" % ((step + 1) * batch_size))
        x_tilde = model(x_train, training=False)
        loss_value = loss_fn(y_train, x_tilde)
        history_metrics["loss_fn_train"].append(loss_value.numpy())

        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in val_dataset:
            x_tilde_val = model(x_batch_val, training=False)
            cokrigLL_metric.update_state(y_batch_val, x_tilde_val)

        val_LL = cokrigLL_metric.result()
        if tf.math.is_nan(val_LL):
            keep_training_flag = False
            break

        history_metrics["metric_LL_val"].append(val_LL.numpy())
        cokrigLL_metric.reset_states()

        if epoch % 10 == 0:
            print("Validation LL over epoch: %.4f" % (float(val_LL)))
            print("Time taken: %.2fs" % (time.time() - start_time))

        if val_LL < best_metric_val:
            model.save_weights(checkpoint_path)
            best_epoch = epoch
            best_metric_val = val_LL

    if not keep_training_flag:
        return
    train_loss = history_metrics["loss_fn_train"][best_epoch]
    val_metric = history_metrics["metric_LL_val"][best_epoch]

    model.load_weights(checkpoint_path)

    _ = cokrigrmse_metric.update_state(
        tf.cast(y_train, dtype=tf.float64),
        tf.cast(model.predict(x_train), dtype=tf.float64),
    )
    train_rmse = cokrigrmse_metric.result().numpy()
    _ = cokrigrmse_metric.update_state(
        tf.cast(y_val, dtype=tf.float64), model.predict(x_val)
    )
    val_rmse = cokrigrmse_metric.result().numpy()
    _ = cokrigrmse_metric.update_state(
        tf.cast(y_test, dtype=tf.float64), model.predict(x_test)
    )
    test_rmse = cokrigrmse_metric.result().numpy()

    _ = cokrigLL_metric.update_state(
        tf.cast(y_train, dtype=tf.float64),
        tf.cast(model.predict(x_train), dtype=tf.float64),
    )
    train_LL = cokrigLL_metric.result().numpy()
    _ = cokrigLL_metric.update_state(
        tf.cast(y_val, dtype=tf.float64), model.predict(x_val)
    )
    val_LL = cokrigLL_metric.result().numpy()
    _ = cokrigLL_metric.update_state(
        tf.cast(y_test, dtype=tf.float64), model.predict(x_test)
    )
    test_LL = cokrigLL_metric.result().numpy()

    # Store hyper-params ad metrics in a params.json file
    param_grid = {
        "num_layers": num_layers,
        "num_nodes": num_nodes,
        "num_outputs": num_outputs,
        "use_dropout": str(use_dropout),
        "drop_rate": drop_rate,
        "use_minibatch": str(use_minibatch),
        "batch_size": batch_size,
        "noise_var": noise_var,
        "corr_coef": corr_coef,
        "learning_rate": learning_rate,
        "l2_reg": l2_reg,
        "checkpoint_path": checkpoint_path,
        "tf_keras_seed": int(random_seed),
        "best_epoch": best_epoch,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
        "test_rmse": test_rmse,
        "train_LL": train_LL,
        "val_LL": val_LL,
        "test_LL": test_LL,
    }
    with open(model_output_path + "/params.json", "w") as f:
        json.dump(param_grid, f, cls=NpEncoder, indent=2)

    with open(model_output_path + "/history.pickle", "wb") as f:
        pickle.dump(history_metrics, f, protocol=pickle.HIGHEST_PROTOCOL)
