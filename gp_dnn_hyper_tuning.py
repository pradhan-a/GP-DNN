from cmath import isnan
import numpy as np
import os
import scipy
import scipy.stats as st
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
from copy import deepcopy
import shutil
import json
import pickle
import time
import random

from utils_dnn import *

tf.keras.backend.set_floatx('float64')

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(101)#101,222,791...,,111,225

data_dir='/scratch/users/pradhan1/GP/Data/'
head_dir='/scratch/users/pradhan1/GP/Tests/'


############# Data loading
with open(data_dir+'training_data_CV_float64_nscored_Jun23_with_topoTexture.pickle', 'rb') as file_pi:
    training_data=pickle.load(file_pi)

x_train=training_data['x_train']
y_train=training_data['y_train']

x_val=training_data['x_val']
y_val=training_data['y_val']

x_test=training_data['x_test']
y_test=training_data['y_test']
###############################


####################
num_inputs=x_train.shape[-1]
num_samp=5000
num_epochs=100
num_coef=4
eval_var=[i for i in range(num_coef)]
######################

########################### Sample hyper-parameters


num_layers_samp=np.random.randint(1,4, size=num_samp)
num_nodes_samp=np.random.randint(30,130, size=num_samp)
num_outputs_samp=np.random.randint(1,30, size=num_samp)

use_dropout_samp=np.random.randint(2,size=num_samp).astype(bool)
droprates_samp=0.01+(0.30-0.01)*np.random.rand(num_samp)

use_minibatch_samp=np.random.randint(1,size=num_samp).astype(bool)##not using mbatchnorm if 1 upper limit in np.random.randint()
minibatch_size_prior=[32,64,128,256,512,1024]
minibatch_size_samp=[minibatch_size_prior[i] 
                            for i in np.random.randint(0,high=len(minibatch_size_prior), size=num_samp)]

noise_vars=[[0.2,0.5],[0.2,0.7],[0.4,0.98],[0.4,0.98]]
noise_var_samp=[noise_vars[i][0]+(noise_vars[i][1]-noise_vars[i][0])*np.random.rand(num_samp) 
							for i in range(num_coef)]

corr_coefs=[[[-0.7,-0.25],[0.01,0.2],[-0.4,-0.05]],[[-0.15,0.15],[-0.05,0.15]],[[-0.9,-0.4]]]
corr_coef_samp=[[cc2[0]+(cc2[1]-cc2[0])*np.random.rand(num_samp) for cc2 in cc1] for cc1 in corr_coefs]

l2_reg_samp=st.loguniform.rvs(0.001,10,size=num_samp)
learning_rate_samp=st.loguniform.rvs(0.001,0.5,size=num_samp)
random_seed_samp=np.random.randint(1e3,1e5,size=num_samp)
############################


############################### DL training
for j in range(num_samp):
    keep_training_flag = True
    #tf.keras.utils.set_random_seed(int(random_seed_samp[j]))
    random.seed(int(random_seed_samp[j]))
    np.random.seed(int(random_seed_samp[j])+1)
    tf.random.set_seed(int(random_seed_samp[j])+2)

    time_s=time.time()

    #Make new directory for storing models 
    checkpoint_dir=head_dir+'model_'+str(j)
    if os.path.isdir(checkpoint_dir):
        shutil.rmtree(checkpoint_dir,ignore_errors=True)
        #continue
    os.mkdir(checkpoint_dir)

    checkpoint_path=checkpoint_dir+'/model_ckpt.ckpt'
    ######## Hyper-parms
    
    num_layers=num_layers_samp[j]
    num_nodes=num_nodes_samp[j]
    num_outputs=num_outputs_samp[j]


    corr_coef=np.eye(num_coef)*0.5
    for k1 in range(num_coef-1):
        for k2 in range(k1+1,num_coef):
            corr_coef[k1,k2]=corr_coef_samp[k1][k2-1-k1][j]
    corr_coef=corr_coef+np.transpose(corr_coef)
    corr_coef=corr_coef.astype('float64')

    noise_var=[nv[j] for nv in noise_var_samp]


    learning_rate=learning_rate_samp[j]
    l2_reg=l2_reg_samp[j]
    ###########################
    
    ################## Functional API for DNN

    #mirrored_strategy = tf.distribute.MirroredStrategy()

    #with mirrored_strategy.scope():
    inputs=tf.keras.Input(shape=(num_inputs,))
    
    for k in range(num_layers):
        if k==0:
            x=tf.keras.layers.Dense(num_nodes,kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(inputs)
        else:
            x=tf.keras.layers.Dense(num_nodes,kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)

        if use_minibatch_samp[j]:
            x=tf.keras.layers.BatchNormalization()(x)
            batch_size=minibatch_size_samp[j]
        else:
            batch_size=x_train.shape[0]

        x=tf.keras.layers.Activation('relu')(x)

        if use_dropout_samp[j]:
            drop_rate=droprates_samp[j]
        else:
            drop_rate=0.0
        x=tf.keras.layers.Dropout(rate=drop_rate)(x)

    outputs=tf.keras.layers.Dense(num_outputs,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    model=tf.keras.Model(inputs=inputs,outputs=outputs)

    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn=GP_marginal_neglikelihood_loss(noise_var=noise_var,corr_coef=corr_coef)
    cokrigr2_metric=cokriging_R2(model,x_train=x_train,y_train_obs=y_train,noise_var=noise_var,
                                     corr_coef=corr_coef,eval_var=eval_var,name='cokrig_R2')
    cokrigrmse_metric=cokriging_RMSE(model,x_train=x_train,y_train_obs=y_train,noise_var=noise_var,
                                     corr_coef=corr_coef,eval_var=eval_var,name='cokrig_RMSE')
    cokrigLL_metric=cokriging_likelihood(model,x_train=x_train,y_train_obs=y_train,noise_var=noise_var,
                                     corr_coef=corr_coef,eval_var=eval_var,name='cokrig_LL')
    #############

    ############## Prepare the training dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=batch_size).batch(batch_size)

    # Prepare the validation dataset.
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(batch_size)
    ####################


    param_grid={ 'num_layers':num_layers,
                  'num_nodes':num_nodes,
                    'num_outputs':num_outputs,
                    'use_dropout':str(use_dropout_samp[j]),
                    'drop_rate':drop_rate,
                    'use_minibatch':str(use_minibatch_samp[j]),
                    'batch_size':batch_size,
                    'noise_var':noise_var,
                    'corr_coef':corr_coef,
                    'learning_rate':learning_rate,
                    'l2_reg':l2_reg,
                    'checkpoint_path':checkpoint_path,
                    'tf_keras_seed':int(random_seed_samp[j])
                }
        # Store hyper-params in a params.json file
    with open(checkpoint_dir+'/params.json', 'w') as f:
        json.dump(param_grid, f,cls=NpEncoder,indent=2)
    ###########

    history_metrics={
        'loss_fn_train':[],
        'metric_LL_val':[]
        
    }
    best_metric_val=np.inf

    print('\nTraining model# '+str(j))
    
    for epoch in range(num_epochs):
        start_time = time.time()

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                x_tilde = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, x_tilde)
            
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Log every 200 batches.
            if (epoch % 10==0) & (step % 2 == 0):
                print("\n Training epoch %d" % (epoch,))
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %d samples" % ((step + 1) * batch_size))
        x_tilde = model(x_train, training=False)
        loss_value = loss_fn(y_train, x_tilde)
        history_metrics['loss_fn_train'].append(loss_value.numpy())

        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in val_dataset:
            x_tilde_val = model(x_batch_val, training=False)
            cokrigLL_metric.update_state(y_batch_val, x_tilde_val)
            
        val_LL = cokrigLL_metric.result()
        if tf.math.is_nan(val_LL):
            keep_training_flag=False
            break

        history_metrics['metric_LL_val'].append(val_LL.numpy())
        cokrigLL_metric.reset_states()
        
        if (epoch % 10==0):
            print("Validation LL over epoch: %.4f" % 
                  (float(val_LL)))
            print("Time taken: %.2fs" % (time.time() - start_time))
        
        if val_LL<best_metric_val:
            model.save_weights(checkpoint_path)
            best_epoch=epoch
            best_metric_val=val_LL
    
    if not keep_training_flag:
        continue
    train_loss=history_metrics['loss_fn_train'][best_epoch]
    val_metric=history_metrics['metric_LL_val'][best_epoch]
    
    model.load_weights(checkpoint_path)

    _=cokrigr2_metric.update_state(tf.cast(y_train,dtype=tf.float64),tf.cast(model.predict(x_train),dtype=tf.float64))
    train_r2=cokrigr2_metric.result().numpy()
    _=cokrigr2_metric.update_state(tf.cast(y_val,dtype=tf.float64),model.predict(x_val))
    val_r2=cokrigr2_metric.result().numpy()
    _=cokrigr2_metric.update_state(tf.cast(y_test,dtype=tf.float64),model.predict(x_test))
    test_r2=cokrigr2_metric.result().numpy()


    _=cokrigrmse_metric.update_state(tf.cast(y_train,dtype=tf.float64),tf.cast(model.predict(x_train),dtype=tf.float64))
    train_rmse=cokrigrmse_metric.result().numpy()
    _=cokrigrmse_metric.update_state(tf.cast(y_val,dtype=tf.float64),model.predict(x_val))
    val_rmse=cokrigrmse_metric.result().numpy()
    _=cokrigrmse_metric.update_state(tf.cast(y_test,dtype=tf.float64),model.predict(x_test))
    test_rmse=cokrigrmse_metric.result().numpy()

    _=cokrigLL_metric.update_state(tf.cast(y_train,dtype=tf.float64),tf.cast(model.predict(x_train),dtype=tf.float64))
    train_LL=cokrigLL_metric.result().numpy()
    _=cokrigLL_metric.update_state(tf.cast(y_val,dtype=tf.float64),model.predict(x_val))
    val_LL=cokrigLL_metric.result().numpy()
    _=cokrigLL_metric.update_state(tf.cast(y_test,dtype=tf.float64),model.predict(x_test))
    test_LL=cokrigLL_metric.result().numpy()


    ## create param grid based on sampled hyper-params
    param_grid={ 'num_layers':num_layers,
              'num_nodes':num_nodes,
                'num_outputs':num_outputs,
                'use_dropout':str(use_dropout_samp[j]),
                'drop_rate':drop_rate,
                'use_minibatch':str(use_minibatch_samp[j]),
                'batch_size':batch_size,
                'noise_var':noise_var,
                'corr_coef':corr_coef,
                'learning_rate':learning_rate,
                'l2_reg':l2_reg,
                'checkpoint_path':checkpoint_path,
                'tf_keras_seed':int(random_seed_samp[j]),
                'best_epoch':best_epoch,
                'train_r2':train_r2,
                'val_r2':val_r2,
                'test_r2':test_r2,
                'train_rmse':train_rmse,
                'val_rmse':val_rmse,
                'test_rmse':test_rmse,
                'train_LL':train_LL,
                'val_LL':val_LL,
                'test_LL':test_LL
            }
    # Store hyper-params in a params.json file
    with open(checkpoint_dir+'/params.json', 'w') as f:
        json.dump(param_grid, f,cls=NpEncoder,indent=2)
    
    with open(checkpoint_dir+'/history.pickle', 'wb') as f:
        pickle.dump(history_metrics, f, protocol=pickle.HIGHEST_PROTOCOL)

    time_e=time.time()

    print(time_e-time_s)
