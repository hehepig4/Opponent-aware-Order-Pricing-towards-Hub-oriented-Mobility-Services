import pandas as pd
import numpy as np
tri_result=pd.read_hdf('tri_result.h5')

nn_result=pd.read_hdf('testdata.h5')
real_data=np.array(tri_result[['quotation'+str(i) for i in range(10)]].values)
model_pred_min={
    'linear':np.array(tri_result[['linear_min']].values),
    'poly':np.array(tri_result[['poly_min']].values),
    'nn_mse':np.array(nn_result[['MSE_loss_min']].values),
    'nn_mse+rank':np.array(nn_result[['MSE_loss_Rank_loss_min']].values),
    'nn_pinball+rank':np.array(nn_result[['pinball_loss_Rank_loss_min']].values),
    'nn_mse+weighted_rank':np.array(nn_result[['weighted_pinball_loss_Rank_loss_min']].values)   
}
model_pred_mid={
    'linear':np.array(tri_result[['linear_mid']].values),
    'poly':np.array(tri_result[['poly_mid']].values),
    'nn_mse':np.array(nn_result[['MSE_loss_mean']].values),
    'nn_mse+rank':np.array(nn_result[['MSE_loss_Rank_loss_mean']].values),
    'nn_pinball+rank':np.array(nn_result[['pinball_loss_Rank_loss_mean']].values),
    'nn_mse+weighted_rank':np.array(nn_result[['weighted_pinball_loss_Rank_loss_mean']].values)   
}
model_pred_max={
    'linear':np.array(tri_result[['linear_max']].values),
    'poly':np.array(tri_result[['poly_max']].values),
    'nn_mse':np.array(nn_result[['MSE_loss_max']].values),
    'nn_mse+rank':np.array(nn_result[['MSE_loss_Rank_loss_max']].values),
    'nn_pinball+rank':np.array(nn_result[['pinball_loss_Rank_loss_max']].values),
    'nn_mse+weighted_rank':np.array(nn_result[['weighted_pinball_loss_Rank_loss_max']].values)   
}
model_pred={
    'min':model_pred_min,
    'mid':model_pred_mid,
    'max':model_pred_max
}
model_name=['linear','poly','nn_mse','nn_mse+rank','nn_pinball+rank','nn_mse+weighted_rank']
picp={'linear':0,'poly':0,'nn_mse':0,'nn_mse+rank':0,'nn_pinball+rank':0,'nn_mse+weighted_rank':0}
pinaw={'linear':0,'poly':0,'nn_mse':0,'nn_mse+rank':0,'nn_pinball+rank':0,'nn_mse+weighted_rank':0}
cwc={'linear':0,'poly':0,'nn_mse':0,'nn_mse+rank':0,'nn_pinball+rank':0,'nn_mse+weighted_rank':0}
real_min_rate={'linear':0,'poly':0,'nn_mse':0,'nn_mse+rank':0,'nn_pinball+rank':0,'nn_mse+weighted_rank':0}
real_max_rate={'linear':0,'poly':0,'nn_mse':0,'nn_mse+rank':0,'nn_pinball+rank':0,'nn_mse+weighted_rank':0}
min_mse={'linear':0,'poly':0,'nn_mse':0,'nn_mse+rank':0,'nn_pinball+rank':0,'nn_mse+weighted_rank':0}
mid_mse={'linear':0,'poly':0,'nn_mse':0,'nn_mse+rank':0,'nn_pinball+rank':0,'nn_mse+weighted_rank':0}
max_mse={'linear':0,'poly':0,'nn_mse':0,'nn_mse+rank':0,'nn_pinball+rank':0,'nn_mse+weighted_rank':0}
alpha=[0.05,0.1,0.15]

for a in alpha:
    print('alpha:',a,file=open('eval_result.txt','a'))
    resultDf=pd.DataFrame(data=np.zeros(shape=(len(model_name),8)),columns=['picp','pinaw','cwc','real_min_rate','real_max_rate','min_mse','mid_mse','max_mse'],index=model_name)
    for n in model_name:
        real_min_rate[n]=0
        real_max_rate[n]=0
        picp[n]=0
        pinaw[n]=0
        cwc[n]=0
        real_min_rate[n]=0
        real_max_rate[n]=0
        min_mse[n]=0
        mid_mse[n]=0
        max_mse[n]=0
    for n in model_name:
        for i in range(10):
            if_smaller=real_data[:,i]<=model_pred['max'][n].T
            if_bigger=real_data[:,i]>=model_pred['min'][n].T
            ok=if_smaller&if_bigger
            ok=ok.astype(int)
            picp[n]+=np.sum(ok)
            if i==0:
                real_min_rate[n]+=np.sum(ok)
                min_mse[n]+=0.5*np.sum(np.square(real_data[:,i]-model_pred['min'][n].T))
            if i==9:
                real_max_rate[n]+=np.sum(ok)
                max_mse[n]+=0.5*np.sum(np.square(real_data[:,i]-model_pred['max'][n].T))
        
        mid_mse[n]+=0.5*np.sum(np.square(0.5*(real_data[:,4]+real_data[:,5])-model_pred['mid'][n].T))    
        
        target_length=real_data[:,-1]-real_data[:,0]
        pred_length=model_pred['max'][n].T-model_pred['min'][n].T
        pinaw[n]+=np.sum(pred_length/target_length)
        
        min_mse[n]=min_mse[n]/real_data.shape[0]
        max_mse[n]=max_mse[n]/real_data.shape[0]
        mid_mse[n]=mid_mse[n]/real_data.shape[0]
        real_min_rate[n]=real_min_rate[n]/real_data.shape[0]
        real_max_rate[n]=real_max_rate[n]/real_data.shape[0]
        picp[n]=picp[n]/(real_data.shape[0]*real_data.shape[1])
        pinaw[n]=pinaw[n]/real_data.shape[0]
        mul=1-a
        gamma=0
        if picp[n]<mul:
            gamma=1
        eta=1
        cwc[n]=pinaw[n]*(1+gamma*np.exp(-eta*(picp[n]-mul)))
        resultDf.loc[n]['picp']=picp[n]
        resultDf.loc[n]['pinaw']=pinaw[n]
        resultDf.loc[n]['cwc']=cwc[n]
        resultDf.loc[n]['real_min_rate']=real_min_rate[n]
        resultDf.loc[n]['real_max_rate']=real_max_rate[n]
        resultDf.loc[n]['min_mse']=min_mse[n]
        resultDf.loc[n]['mid_mse']=mid_mse[n]
        resultDf.loc[n]['max_mse']=max_mse[n]
    print(resultDf,'\n',file=open('eval_result.txt','a'))