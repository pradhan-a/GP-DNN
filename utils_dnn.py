import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import json

class GP_marginal_neglikelihood_loss(tf.keras.losses.Loss):
    
    def __init__(self,noise_var=[0.5,0.1],corr_coef=np.array([[1,0.3],[0.3,1]])):
        super().__init__()
        self.num_var=len(noise_var)
        self.noise_var=noise_var
        self.corr_coef=corr_coef
    
    def call(self, y_true, x_tilde_pred):
        
        y_true_cokrig=tf.concat([tf.expand_dims(tf.gather(y_true[:,0],
                                tf.squeeze(tf.where(y_true[:,-1]==i)),axis=0),axis=-1) 
                                 for i in range(self.num_var)],axis=0)
        
        x_tilde_pred_cokrig=[tf.gather(x_tilde_pred,tf.squeeze(tf.where(y_true[:,-1]==i)),axis=0) 
                             for i in range(self.num_var)]


        x_tilde_pred_cov=cokriging_covariance(x_tilde_pred_cokrig,x_tilde_pred_cokrig,is_training_cov=True)
        x_tilde_pred_cov = tf.concat([tf.concat(x_tilde_pred_cov[i],axis=-1) for i in range(len(x_tilde_pred_cov))],axis=0)
    
        
        data_fit=0.5*tf.linalg.matmul(tf.linalg.matmul(y_true_cokrig,tf.linalg.inv(x_tilde_pred_cov),transpose_a=True),y_true_cokrig)
        _,complex_logdet=tf.linalg.slogdet(x_tilde_pred_cov)
        complexity=0.5*complex_logdet
        
        return tf.squeeze(data_fit+complexity)

class cokriging_RMSE(tf.keras.metrics.Metric):
    def __init__(self,model,x_train,y_train_obs,noise_var=[0.5,0.1],corr_coef=np.array([[1,0.3],[0.3,1]]),eval_var=[0],name='cokriging_RMSE',**kwargs):
        super(cokriging_RMSE,self).__init__(name=name, **kwargs)
        self.model=model
        self.x_train=x_train
        self.y_train_obs=y_train_obs
        self.num_var=len(noise_var)
        self.noise_var=noise_var
        self.corr_coef=corr_coef
        self.eval_var=eval_var
        self.krig_RMSE=tf.Variable(-999)
        
    def update_state(self, y_true, x_tilde_pred, sample_weight=None):
        
        ###### For user-provided x_train 
        x_tilde_train=self.model(self.x_train)
        
        x_tilde_train_cokrig=[tf.gather(x_tilde_train,
                                tf.squeeze(tf.where(self.y_train_obs[:,-1]==i)),axis=0)
                                for i in range(self.num_var)]
        
        y_train=tf.concat([tf.expand_dims(tf.gather(self.y_train_obs[:,0],
                                tf.squeeze(tf.where(self.y_train_obs[:,-1]==i)),axis=0),axis=-1) 
                                for i in range(self.num_var)],axis=0)
        
        x_tilde_train_cov=self.cokriging_covariance(x_tilde_train_cokrig,x_tilde_train_cokrig,is_training_cov=True)
        
        x_tilde_train_cov=tf.concat([tf.concat(x_tilde_train_cov[i],axis=-1) 
                                    for i in range(len(x_tilde_train_cov))],axis=0)
        
        ########
        
        ###### For tf-provided x_tilde
        y_true_cokrig=[tf.expand_dims(tf.gather(y_true[:,0],
                                tf.squeeze(tf.where(y_true[:,-1]==i)),axis=0),axis=-1)
                                 for i in range(self.num_var)]
        
        x_tilde_pred_cokrig=[tf.gather(x_tilde_pred,
                                tf.squeeze(tf.where(y_true[:,-1]==i)),axis=0) for i in range(self.num_var)]

        x_tilde_pred_cov=cokriging_covariance(x_tilde_pred_cokrig,x_tilde_train_cokrig,is_training_cov=False)
        
        
        ######
        rmses=[]
        for i in self.eval_var:
            
            x_tilde_pred_cov1=tf.concat(x_tilde_pred_cov[i],axis=-1)
   
            krig_weights=tf.linalg.solve(x_tilde_train_cov,tf.transpose(x_tilde_pred_cov1))

            y_pred=tf.linalg.matmul(tf.transpose(krig_weights),y_train)
            
            rmses.append(tf.sqrt(tf.reduce_mean((y_pred-y_true_cokrig[i])**2)))
        
        
        self.krig_RMSE=tf.reduce_mean(rmses)
        ######
    
    def result(self):
        return self.krig_RMSE

class cokriging_R2(tf.keras.metrics.Metric):
    def __init__(self,model,x_train,y_train_obs,noise_var=[0.5,0.1],corr_coef=np.array([[1,0.3],[0.3,1]]),eval_var=[0],name='cokriging_R2',**kwargs):
        super(cokriging_R2,self).__init__(name=name, **kwargs)
        self.model=model
        self.x_train=x_train
        self.y_train_obs=y_train_obs
        self.num_var=len(noise_var)
        self.noise_var=noise_var
        self.corr_coef=corr_coef
        self.eval_var=eval_var
        self.krig_R2=tf.Variable(-999)
        
    def update_state(self, y_true, x_tilde_pred, sample_weight=None):
        
        ###### For user-provided x_train 
        x_tilde_train=self.model(self.x_train)
        
        x_tilde_train_cokrig=[tf.gather(x_tilde_train,
                                tf.squeeze(tf.where(self.y_train_obs[:,-1]==i)),axis=0)
                                for i in range(self.num_var)]
        
        y_train=tf.concat([tf.expand_dims(tf.gather(self.y_train_obs[:,0],
                                tf.squeeze(tf.where(self.y_train_obs[:,-1]==i)),axis=0),axis=-1) 
                                for i in range(self.num_var)],axis=0)
        
        x_tilde_train_cov=self.cokriging_covariance(x_tilde_train_cokrig,x_tilde_train_cokrig,is_training_cov=True)
        
        x_tilde_train_cov=tf.concat([tf.concat(x_tilde_train_cov[i],axis=-1) 
                                    for i in range(len(x_tilde_train_cov))],axis=0)
        
        ########
        
        ###### For tf-provided x_tilde
        y_true_cokrig=[tf.expand_dims(tf.gather(y_true[:,0],
                                tf.squeeze(tf.where(y_true[:,-1]==i)),axis=0),axis=-1)
                                 for i in range(self.num_var)]
        
        x_tilde_pred_cokrig=[tf.gather(x_tilde_pred,
                                tf.squeeze(tf.where(y_true[:,-1]==i)),axis=0) for i in range(self.num_var)]

        x_tilde_pred_cov=self.cokriging_covariance(x_tilde_pred_cokrig,x_tilde_train_cokrig,is_training_cov=False)
        
        
        ######
        R2s=[]
        for i in self.eval_var:
            
            x_tilde_pred_cov1=tf.concat(x_tilde_pred_cov[i],axis=-1)
   
            krig_weights=tf.linalg.solve(x_tilde_train_cov,tf.transpose(x_tilde_pred_cov1))

            y_pred=tf.linalg.matmul(tf.transpose(krig_weights),y_train)
            
            R2s.append(1-tf.reduce_sum((y_true_cokrig[i]-y_pred)**2)
                       /tf.reduce_sum((y_true_cokrig[i]-tf.reduce_mean(y_true_cokrig[i]))**2))
        
        
        self.krig_R2=tf.reduce_mean(R2s)
        ######
    
    def result(self):
        return self.krig_R2

class cokriging_likelihood(tf.keras.metrics.Metric):
    def __init__(self,model,x_train,y_train_obs,noise_var=[0.5,0.1],corr_coef=np.array([[1,0.3],[0.3,1]]),eval_var=[0],name='cokriging_likelihood',**kwargs):
        super(cokriging_likelihood,self).__init__(name=name, **kwargs)
        self.model=model
        self.x_train=x_train
        self.y_train_obs=y_train_obs
        self.num_var=len(noise_var)
        self.noise_var=noise_var
        self.corr_coef=corr_coef
        self.eval_var=eval_var
        self.krig_likelihood=tf.Variable(-999)
        
    def update_state(self, y_true, x_tilde_pred, sample_weight=None):
        
        ###### For user-provided x_train 
        x_tilde_train=self.model(self.x_train)
        
        x_tilde_train_cokrig=[tf.gather(x_tilde_train,
                                tf.squeeze(tf.where(self.y_train_obs[:,-1]==i)),axis=0)
                                for i in range(self.num_var)]
        
        y_train=tf.concat([tf.expand_dims(tf.gather(self.y_train_obs[:,0],
                                tf.squeeze(tf.where(self.y_train_obs[:,-1]==i)),axis=0),axis=-1) 
                                for i in range(self.num_var)],axis=0)
        
        x_tilde_train_cov=self.cokriging_covariance(x_tilde_train_cokrig,x_tilde_train_cokrig,is_training_cov=True)
        
        x_tilde_train_cov=tf.concat([tf.concat(x_tilde_train_cov[i],axis=-1) 
                                    for i in range(len(x_tilde_train_cov))],axis=0)
        
        ########
        
        ###### For tf-provided x_tilde
        y_true_cokrig=tf.concat([tf.expand_dims(tf.gather(y_true[:,0],
                                tf.squeeze(tf.where(y_true[:,-1]==i)),axis=0),axis=-1)
                                 for i in range(self.num_var)],axis=0)
        
        x_tilde_pred_cokrig=[tf.gather(x_tilde_pred,
                                tf.squeeze(tf.where(y_true[:,-1]==i)),axis=0) for i in range(self.num_var)]

        x_tilde_pred_cov=self.cokriging_covariance(x_tilde_pred_cokrig,x_tilde_train_cokrig,is_training_cov=False)
        
        
        x_tilde_pred_pred_cov=self.cokriging_covariance(x_tilde_pred_cokrig,x_tilde_pred_cokrig,is_training_cov=False)


        ####
        x_tilde_pred_cov=tf.concat([tf.concat(x_tilde_pred_cov[i],axis=-1) for i in range(self.num_var)],axis=0)
        x_tilde_pred_pred_cov=tf.concat([tf.concat(x_tilde_pred_pred_cov[i],axis=-1) for i in range(self.num_var)],axis=0)

        x_tilde_train_cov_inv=tf.linalg.inv(x_tilde_train_cov)
        y_pred=tf.linalg.matmul(tf.linalg.matmul(x_tilde_pred_cov,x_tilde_train_cov_inv),y_train)
        y_cov=x_tilde_pred_pred_cov-tf.linalg.matmul(
                                                    tf.linalg.matmul(x_tilde_pred_cov,x_tilde_train_cov_inv),
                                                    x_tilde_pred_cov,transpose_b=True)
        y_cov=y_cov+tf.eye(tf.shape(y_cov)[0],dtype=tf.float64)*1e-6

        try:
            self.krig_likelihood=-tfp.distributions.MultivariateNormalTriL(
                loc=tf.squeeze(y_pred),scale_tril=tf.linalg.cholesky(y_cov)
            ).log_prob(tf.squeeze(y_true_cokrig))
        except:
            self.krig_likelihood=np.nan
        ######
    
    def result(self):
        return self.krig_likelihood

    

    
def cokriging_covariance(self,x_tilde_pred,x_tilde_train,is_training_cov):
    
    x_tilde_cov=[[None for j in range(self.num_var)] for i in range(self.num_var)]

    for i in range(self.num_var):
        for j in range(i+1):
            if i==j:
                x_tilde_cov[i][j]=tfp.math.psd_kernels.MaternFiveHalves().matrix(x_tilde_pred[i],x_tilde_train[j])
                if is_training_cov:
                    x_tilde_cov_diag=tf.linalg.diag_part(x_tilde_cov[i][j])
                    x_tilde_cov[i][j]=tf.linalg.set_diag(x_tilde_cov[i][j],x_tilde_cov_diag+self.noise_var[i]*tf.ones_like(x_tilde_cov_diag))

            else:
                x_tilde_cov[i][j]=tf.sign(self.corr_coef[i,j])*tfp.math.psd_kernels.MaternFiveHalves(amplitude=
                                tf.sqrt(tf.abs(self.corr_coef[i,j]))).matrix(x_tilde_pred[i],x_tilde_train[j])

    for i in range(self.num_var-1):
        for j in range(i+1,self.num_var):
            if is_training_cov:
                x_tilde_cov[i][j]=tf.transpose(x_tilde_cov[j][i])
            else:
                x_tilde_cov[i][j]=tf.sign(self.corr_coef[i,j])*tfp.math.psd_kernels.MaternFiveHalves(amplitude=
                                tf.sqrt(tf.abs(self.corr_coef[i,j]))).matrix(x_tilde_pred[i],x_tilde_train[j])

            
    return tf.concat([tf.concat(x_tilde_cov[i],axis=-1) for i in range(len(x_tilde_cov))],axis=0)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
