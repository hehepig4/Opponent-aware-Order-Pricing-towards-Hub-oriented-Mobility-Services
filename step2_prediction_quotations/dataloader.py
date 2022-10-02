
import torch
import torch.utils.data as pytorch_data
import numpy as np
import configure
import pandas as pd
def _init():
    global _global_dict
    try:
        _global_dict
    except:
        _global_dict = {}
def exist_key(key):
    return key in _global_dict
def set_value(key,value):
    try:
        _global_dict[key]=value
    except:
        _init()
        _global_dict[key]=value
def get_value(key):
    try:
        return _global_dict[key]
    except:
        if key=='file':
            set_value(key,pd.read_hdf('data.h5','data'))
        return _global_dict[key]
def read(s=0,e=None):
    data=get_value('file')
    
    #print(data.shape)
    return data.iloc[s:e]

min_la=configure.min_la
max_la=configure.max_la
min_lo=configure.min_lo
max_lo=configure.max_lo
step_dis=step_dis=configure.step_dis

def as_tensor(pickup_latitude,pickup_longitude,dropoff_latitude,dropoff_longitude,weekday,minutes_in_day,trip_time_in_secs,trip_distance):
    start_block_la=int(np.round((pickup_latitude-min_la)/step_dis))
    start_block_lo=int(np.round((pickup_longitude-min_lo)/step_dis))
    end_block_la=int(np.round((dropoff_latitude-min_la)/step_dis))
    end_block_lo=int(np.round((dropoff_longitude-min_lo)/step_dis))
    vector_angle=np.arctan2(dropoff_latitude-pickup_latitude,dropoff_longitude-pickup_longitude)
    matrix=np.zeros(shape=((round((max_la-min_la)/step_dis)+1,round((max_lo-min_lo)/step_dis)+1,2)),dtype=np.float32)
    matrix[start_block_la,start_block_lo,0]=1
    matrix[end_block_la,end_block_lo,1]=1
    vector_matrix=np.ones(shape=((round((max_la-min_la)/step_dis)+1,round((max_lo-min_lo)/step_dis)+1,1)),dtype=np.float32)
    vector_matrix*=vector_angle
    minutes_matrix=np.ones(shape=((round((max_la-min_la)/step_dis)+1,round((max_lo-min_lo)/step_dis)+1,1)),dtype=np.float32)
    minutes_matrix*=minutes_in_day
    trip_time_matrix=np.ones(shape=((round((max_la-min_la)/step_dis)+1,round((max_lo-min_lo)/step_dis)+1,1)),dtype=np.float32)
    trip_time_matrix*=trip_time_in_secs
    trip_distance_matrix=np.ones(shape=((round((max_la-min_la)/step_dis)+1,round((max_lo-min_lo)/step_dis)+1,1)),dtype=np.float32)
    trip_distance_matrix*=trip_distance
    weekday_matrix=np.ones(shape=((round((max_la-min_la)/step_dis)+1,round((max_lo-min_lo)/step_dis)+1,1)),dtype=np.float32)
    weekday_matrix*=weekday
    matrix=np.concatenate((matrix,vector_matrix,minutes_matrix,trip_time_matrix,trip_distance_matrix,weekday_matrix),axis=2)
    matrix=matrix.swapaxes(0,2)
    tens=torch.from_numpy(matrix)
    
    return tens

def as_tensor2(pickup_latitude,pickup_longitude,dropoff_latitude,dropoff_longitude,weekday,minutes_in_day,trip_time_in_secs,trip_distance):
    start_block_la=int(np.round((pickup_latitude-min_la)/step_dis))
    start_block_lo=int(np.round((pickup_longitude-min_lo)/step_dis))
    end_block_la=int(np.round((dropoff_latitude-min_la)/step_dis))
    end_block_lo=int(np.round((dropoff_longitude-min_lo)/step_dis))
    matrix=np.zeros(shape=((2,round((max_la-min_la)/step_dis)+1,round((max_lo-min_lo)/step_dis)+1)),dtype=np.float32)
    matrix[0,start_block_la,start_block_lo]=1
    matrix[1,end_block_la,end_block_lo]=1
    return torch.from_numpy(matrix)



from tqdm import tqdm
from multiprocessing.pool import ThreadPool as Pool



class streamingdataset1(pytorch_data.Dataset):
    '''
    output: x=[batch,seq_len,]
            y=[tensor(quantiles)[quantiles_num]]
    '''
    def __init__(self,start=0,end=None,seq_len=configure.seq_len):
        super(streamingdataset1,self).__init__()
        self.oridata=read(start,end)
        self.seq_len=seq_len
        self.length=(end-start)//seq_len
        self.start=start//self.seq_len #!bug
        self.end=end//self.seq_len
            
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
        #y['0per']=min(data[['fare_amount','func1','func2','func3','func4','func5','func6','func7','func8','func9']].values)
        for i in configure.percentiles:
            #y[str(i)+'per']=np.percentile(data[['fare_amount','func1','func2','func3','func4','func5','func6','func7','func8','func9']].values,i,axis=1)
            y['min']=np.min(data[['fare_amount','func1','func2','func3','func4','func5','func6','func7','func8','func9']].values,axis=1)
        return torch.stack(temp),torch.tensor([weekday,minutes_in_day,trip_distance,trip_time_in_secs,vector_angle],dtype=torch.float32),torch.from_numpy(np.array(y.iloc[-1,:],dtype=np.float32))
    def __getitem__(self, index):
        return self._handle_data(self.oridata.iloc[self.start+index*self.seq_len:self.start+(index+1)*self.seq_len])
    def __len__(self):
        return self.length-1
    
    
    
class streamingdataset2(pytorch_data.Dataset):
    '''output:[height,width,channels=9] for cnn
    channels:[seq_len_start_sum,seq_len_end_sum,start_block,end_block,vector_angle,minutes_in_day,trip_time_in_secs,trip_distance,weekday]
    '''
    def __init__(self,start=0,end=None,seq_len=configure.seq_len):
        super(streamingdataset2,self).__init__()
        self.oridata=read(start,end)
        self.seq_len=seq_len
        self.length=(end-start)
        self.start=start
        self.end=end
            
    def _handle_data(self,data):
        x=data[['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude','minutes_in_day','weekday','trip_time_in_secs','trip_distance']].values
        
        base_matrix=as_tensor2(*x[0:4])
        weekday=data['weekday'].values[-1]
        minutes_in_day=data['minutes_in_day'].values[-1]
        trip_time_in_secs=data['trip_time_in_secs'].values[-1]
        trip_distance=data['trip_distance'].values[-1]
        vector_angle=np.arctan2(x[-1][1]-x[-2][1],x[-1][0]-x[-2][0])
        
        
        x=torch.from_numpy(base_matrix)
        x=x.permute(2,0,1)
        
        #y['0per']=min(data[['fare_amount','func1','func2','func3','func4','func5','func6','func7','func8','func9']].values)
        #for i in configure.percentiles:
            #y[str(i)+'per']=np.percentile(data[['fare_amount','func1','func2','func3','func4','func5','func6','func7','func8','func9']].values[-1,:],i,axis=1,interpolation='nearest')
        y=data[['fare_amount','func1','func2','func3','func4','func5','func6','func7','func8','func9']].values[:]
        y=min(y)
        return (x,torch.tensor([weekday,minutes_in_day,trip_time_in_secs,trip_distance,vector_angle])),torch.tensor(y)
    def __getitem__(self, index):
        return self._handle_data(self.oridata.iloc[self.start+index])
    def __len__(self):
        return self.length-1
    
class streamingdataset3(pytorch_data.Dataset):
    '''
    output: x=[batch,seq_len,]
            y=[tensor(quantiles)[quantiles_num]]
    '''
    def __init__(self,start=0,end=None,seq_len=configure.seq_len):
        super(streamingdataset3,self).__init__()
        self.oridata=read(start,end)
        self.seq_len=seq_len
        self.length=(end-start)//seq_len
        self.start=start//self.seq_len #!bug
        self.end=end//self.seq_len
            
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
        #y['0per']=min(data[['fare_amount','func1','func2','func3','func4','func5','func6','func7','func8','func9']].values)
        d2=data[['fare_amount','func1','func2','func3','func4','func5','func6','func7','func8','func9']].values[-1,:]
        y=np.quantile(d2,configure.percentiles,axis=0,interpolation='nearest')
            #y['min']=np.min(data[['fare_amount','func1','func2','func3','func4','func5','func6','func7','func8','func9']].values,axis=1)
        return torch.stack(temp),torch.tensor([weekday,minutes_in_day,trip_distance,trip_time_in_secs,vector_angle],dtype=torch.float32),torch.from_numpy(np.array(y,dtype=np.float32))
    def __getitem__(self, index):
        return self._handle_data(self.oridata.iloc[self.start+index*self.seq_len:self.start+(index+1)*self.seq_len])
    def __len__(self):
        return self.length-1
    
class streamingdataset4(pytorch_data.Dataset):
    '''
    output: x=[batch,seq_len,]
            y=[tensor(quantiles)[quantiles_num]]
    '''
    def __init__(self,start=0,end=None,seq_len=configure.seq_len):
        super(streamingdataset4,self).__init__()
        self.oridata=read(start,end)
        self.seq_len=seq_len
        self.length=(end-start)//seq_len
        self.start=start//self.seq_len #!bug
        self.end=end//self.seq_len
            
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
        #y['0per']=min(data[['fare_amount','func1','func2','func3','func4','func5','func6','func7','func8','func9']].values)
        d2=data[['fare_amount','func1','func2','func3','func4','func5','func6','func7','func8','func9']].values[-1,:]
        y=np.quantile(d2,configure.percentiles,axis=0,interpolation='nearest')
            #y['min']=np.min(data[['fare_amount','func1','func2','func3','func4','func5','func6','func7','func8','func9']].values,axis=1)
        return torch.stack(temp),torch.tensor([weekday,minutes_in_day,trip_distance,trip_time_in_secs,vector_angle],dtype=torch.float32),torch.from_numpy(np.array(y,dtype=np.float32)),torch.from_numpy(np.array(d2,dtype=np.float32)).sort(descending=False)[0]
    def __getitem__(self, index):
        return self._handle_data(self.oridata.iloc[self.start+index*self.seq_len:self.start+(index+1)*self.seq_len])
    def __len__(self):
        return self.length-1
    
streamingdataset=streamingdataset1
class streamingdataset5(pytorch_data.Dataset):
    '''
    output: x=[batch,seq_len,]
            y=[tensor(quantiles)[quantiles_num]]
    '''
    def __init__(self,start=0,end=None,seq_len=configure.seq_len):
        super(streamingdataset5,self).__init__()
        self.oridata=read(start,end)
        self.seq_len=seq_len
        
        self.start=start#!bug
        self.end=end-seq_len
        self.length=(self.end-self.start)
    def _handle_data(self,data):
        x=data[['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude','minutes_in_day','weekday','trip_time_in_secs','trip_distance']].values
        #temp=[]
            #multiprocessing
        # def f(xa):
        #    return as_tensor2(*xa)
        #with Pool(processes=configure.read_nworker) as pool:
        #    for i in pool.imap(f,x):
        #        temp.append(i)
        #y=pd.DataFrame()
        #weekday=data['weekday'].values[-1]
        #minutes_in_day=data['minutes_in_day'].values[-1]
        #trip_time_in_secs=data['trip_time_in_secs'].values[-1]
        #trip_distance=data['trip_distance'].values[-1]
        #vector_angle=np.arctan2(data['dropoff_latitude'].values[-1]-data['pickup_latitude'].values[-1],data['dropoff_longitude'].values[-1]-data['pickup_longitude'].values[-1])
        #y['0per']=min(data[['fare_amount','func1','func2','func3','func4','func5','func6','func7','func8','func9']].values)
        #d2=data[['fare_amount','func1','func2','func3','func4','func5','func6','func7','func8','func9']].values[-1,:]
        #y=np.quantile(d2,configure.percentiles,axis=0,interpolation='nearest')
        fare_amount=data['fare_amount'].values[-1]
            #y['min']=np.min(data[['fare_amount','func1','func2','func3','func4','func5','func6','func7','func8','func9']].values,axis=1)
        #return torch.stack(temp),torch.tensor([weekday,minutes_in_day,trip_distance,trip_time_in_secs,vector_angle],dtype=torch.float32),torch.from_numpy(np.array(y,dtype=np.float32)),torch.from_numpy(np.array(d2,dtype=np.float32)).sort(descending=False)[0],torch.tensor(x[-1,:],dtype=torch.float32)
        return torch.tensor(fare_amount,dtype=torch.float32)
    def __getitem__(self, index):
        return self._handle_data(self.oridata.iloc[self.start+index:self.start+index+self.seq_len])
    def __len__(self):
        return self.length-1

