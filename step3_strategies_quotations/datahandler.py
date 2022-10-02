
import torch
import torch.utils.data as pytorch_data
import numpy as np
import configure
import pandas as pd
import ray
from tqdm import tqdm
from multiprocessing.pool import ThreadPool as Pool
def _init():
    global _global_dict
    global _global_dict_id
    try:
        _global_dict
    except:
        _global_dict = {}
    _global_dict_id=ray.put(_global_dict)
def exist_key(key):
    return key in ray.get(_global_dict_id)
def set_value(key,value):
    try:
        ray.get(_global_dict_id)[key]=value
    except:
        _init()
        ray.get(_global_dict_id)[key]=value
def get_value(key):
    try:
        return ray.get(_global_dict_id)[key]
    except:
        return None
def read(s=0,e=None):
    data=get_value('file')
    
    #print(data.shape)
    return data.iloc[s:e]

min_la=configure.min_la
max_la=configure.max_la
min_lo=configure.min_lo
max_lo=configure.max_lo
step_dis=step_dis=configure.step_dis
def as_tensor2(pickup_latitude,pickup_longitude,dropoff_latitude,dropoff_longitude,weekday,minutes_in_day,trip_time_in_secs,trip_distance):
    start_block_la=int(np.round((pickup_latitude-min_la)/step_dis))
    start_block_lo=int(np.round((pickup_longitude-min_lo)/step_dis))
    end_block_la=int(np.round((dropoff_latitude-min_la)/step_dis))
    end_block_lo=int(np.round((dropoff_longitude-min_lo)/step_dis))
    matrix=np.zeros(shape=((2,round((max_la-min_la)/step_dis)+1,round((max_lo-min_lo)/step_dis)+1)),dtype=np.float32)
    matrix[0,start_block_la,start_block_lo]=1
    matrix[1,end_block_la,end_block_lo]=1
    return torch.from_numpy(matrix).float()

class requesthandler(pytorch_data.Dataset):
    def __init__(self,data):
        if data is not None:
            self.data=data
        else:
            raise Exception('data is None')
    def __getitem__(self,index):
        data=self.data.iloc[index,:][['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude','minutes_in_day','weekday','trip_time_in_secs','trip_distance']]
        matrix=as_tensor2(
            *data.values
        ) 
        y=self.data.iloc[index,:][['quotation'+str(i) for i in range(10)]+['weighted_pinball_loss_Rank_loss_min','weighted_pinball_loss_Rank_loss_mean','weighted_pinball_loss_Rank_loss_max']]
        cost=self.data.iloc[index,:].loc['fare_amonut']
        weekday=data['weekday']
        minutes_in_day=data['minutes_in_day']
        trip_time_in_secs=data['trip_time_in_secs']
        trip_distance=data['trip_distance']
        vector_angle=np.arctan2(data['dropoff_latitude']-data['pickup_latitude'],data['dropoff_longitude']-data['pickup_longitude'])
        other_quos=y[['quotation'+str(i) for i in range(10)]].values
        predic_quos=y[['weighted_pinball_loss_Rank_loss_min','weighted_pinball_loss_Rank_loss_mean','weighted_pinball_loss_Rank_loss_max']].values
        return matrix,torch.tensor([weekday,minutes_in_day,trip_time_in_secs,trip_distance,vector_angle],dtype=torch.float32),torch.from_numpy(other_quos).float(),torch.from_numpy(predic_quos).float(),torch.tensor([cost],dtype=torch.float32)
    def __len__(self):
        return self.data.shape[0]
'''
class streamingdataset5(pytorch_data.Dataset):
    
    def __init__(self,start=0,end=None,seq_len=configure.seq_len):
        super(streamingdataset5,self).__init__()
        self.oridata=read(start,end)
        self.seq_len=seq_len
        
        self.start=start#!bug
        self.end=end-seq_len
        self.length=(self.end-self.start)
    def _handle_data(self,data):
        x=data[['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude','minutes_in_day','weekday','trip_time_in_secs','trip_distance']].values
        temp=[]
            #multiprocessing
        def f(xa):
            return as_tensor2(*xa)
        with Pool(processes=configure.read_nworker) as pool:
            for i in pool.imap(f,x):
                temp.append(i)
        y=pd.DataFrame()
        weekday=data['weekday'].values[-1]
        minutes_in_day=data['minutes_in_day'].values[-1]
        trip_time_in_secs=data['trip_time_in_secs'].values[-1]
        trip_distance=data['trip_distance'].values[-1]
        vector_angle=np.arctan2(data['dropoff_latitude'].values[-1]-data['pickup_latitude'].values[-1],data['dropoff_longitude'].values[-1]-data['pickup_longitude'].values[-1])
        #y['0per']=min(data[['fare_amonut','func1','func2','func3','func4','func5','func6','func7','func8','func9']].values)
        d2=data[['fare_amonut','func1','func2','func3','func4','func5','func6','func7','func8','func9']].values[-1,:]
        y=np.quantile(d2,configure.percentiles,axis=0,interpolation='nearest')
            #y['min']=np.min(data[['fare_amonut','func1','func2','func3','func4','func5','func6','func7','func8','func9']].values,axis=1)
        return torch.stack(temp),torch.tensor([weekday,minutes_in_day,trip_distance,trip_time_in_secs,vector_angle],dtype=torch.float32),torch.from_numpy(np.array(y,dtype=np.float32)),torch.from_numpy(np.array(d2,dtype=np.float32)).sort(descending=False)[0],torch.tensor(x[-1,:],dtype=torch.float32)
    def __getitem__(self, index):
        return self._handle_data(self.oridata.iloc[self.start+index:self.start+index+self.seq_len])
    def __len__(self):
        return self.length-1
'''