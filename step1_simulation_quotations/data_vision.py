
import os
from datetime import datetime

import dask.dataframe as dd
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import time
import configure

train_types1={
    'pickup_datetime': 'str',
    'dropoff_datetime': 'str',
    'trip_time_in_secs': 'float32',
    'trip_distance': 'float32',
    'pickup_longitude': 'float32',
    'pickup_latitude': 'float32',
    'dropoff_longitude': 'float32',
    'dropoff_latitude': 'float32',
}
train_types2={
    ' fare_amount': 'float32',
    ' surcharge': 'float32',
    ' mta_tax': 'float32',
    ' tip_amount': 'float32',
    ' tolls_amount': 'float32',
    ' total_amount': 'float32',
}
def select_within_boundingbox(df, BB):
    return (df.pickup_longitude >= BB[0]) & (df.pickup_longitude <= BB[1]) & \
        (df.pickup_latitude >= BB[2]) & (df.pickup_latitude <= BB[3]) & \
        (df.dropoff_longitude >= BB[0]) & (df.dropoff_longitude <= BB[1]) & \
        (df.dropoff_latitude >= BB[2]) & (df.dropoff_latitude <= BB[3])
        
        
def remove_datapoints_from_water(df):
    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx*(longitude - BB[0])/(BB[1]-BB[0])).astype('int'), \
            (dy - dy*(latitude - BB[2])/(BB[3]-BB[2])).astype('int')

    # define bounding box
    BB = configure.boundingbox
    
    # read nyc mask and turn into boolean map with
    # land = True, water = False
    nyc_mask = plt.imread('nyc_mask-74.5_-72.8_40.5_41.8.png')[:,:,0] > 0.9
    # calculate for each lon,lat coordinate the xy coordinate in the mask map
    pickup_x, pickup_y = lonlat_to_xy(df.pickup_longitude, df.pickup_latitude, 
                                    nyc_mask.shape[1], nyc_mask.shape[0], BB)
    dropoff_x, dropoff_y = lonlat_to_xy(df.dropoff_longitude, df.dropoff_latitude, 
                                    nyc_mask.shape[1], nyc_mask.shape[0], BB)    
    # calculate boolean index
    idx = nyc_mask[pickup_y-1, pickup_x-1] & nyc_mask[dropoff_y-1, dropoff_x-1]
    
    # return only datapoints on land
    return df[idx]


class datahandler:
    
    def __init__(self,filenum=0):
        start=time.time()
        self.count=0
        print("🤖init:data_loader_init... ")
        if filenum != 0:
            print("😊init:reading target file",filenum)
            data_handler = pd.read_csv('./data/trip_data_'+str(filenum)+'.csv', usecols=[5,6,8,9,10,11,12,13],dtype=train_types1)
            fare_handler = pd.read_csv('./data/trip_fare_'+str(filenum)+'.csv', usecols=[5,6,7,8,9,10],dtype=train_types2)
            self.data=pd.concat([data_handler,fare_handler],axis=1)
            self.data.columns=self.data.columns.str.strip()
            self.data.dropna(how='any',axis='rows',inplace=True)
            self.data['pickup_datetime']=pd.to_datetime(self.data['pickup_datetime'],format='%Y-%m-%d %H:%M:%S',errors='coerce')
            self.data['dropoff_datetime']=pd.to_datetime(self.data['dropoff_datetime'],format='%Y-%m-%d %H:%M:%S',errors='coerce')
            self.data['trip_time_in_secs']=pd.to_numeric(self.data['trip_time_in_secs'],errors='coerce',downcast='float')
            self.data.dropna(axis=0,how='any')    
            
        else:
            filenum=1
            data_handler=pd.DataFrame(columns=list(train_types1.keys())+list(train_types2.keys()))
            data_handler.columns=data_handler.columns.str.strip()
            while len(data_handler)<configure.data_max_size:
                if filenum>12 : break
                waiting_num=configure.data_max_size-len(data_handler)
                print("😊init:waiting read data num",waiting_num)
                print("😊init:reading file",filenum)
                temp1=pd.read_csv('./data/trip_data_'+str(filenum)+'.csv', usecols=[5,6,8,9,10,11,12,13],dtype=train_types1)
                temp2=pd.read_csv('./data/trip_fare_'+str(filenum)+'.csv', usecols=[5,6,7,8,9,10],dtype=train_types2)
                temp=pd.concat([temp1,temp2],axis=1)
                del temp1,temp2
                temp.columns=temp.columns.str.strip()
                temp.dropna(axis=0,how='any')
                temp['pickup_datetime']=pd.to_datetime(temp['pickup_datetime'],format='%Y-%m-%d %H:%M:%S',errors='coerce')
                temp['dropoff_datetime']=pd.to_datetime(temp['dropoff_datetime'],format='%Y-%m-%d %H:%M:%S',errors='coerce')
                temp['trip_time_in_secs']=pd.to_numeric(temp['trip_time_in_secs'],errors='coerce',downcast='float')
                temp.dropna(axis=0,how='any')
                data_handler=pd.concat([data_handler,temp],ignore_index=True,axis=0)
                filenum+=1
            self.data=data_handler
        
        
        #data_handler['pickup_datetime']=pd.to_datetime(data_handler['pickup_datetime'],format='%Y-%m-%d %H:%M:%S',errors='coerce')
        #data_handler['dropoff_datetime']=pd.to_datetime(data_handler['dropoff_datetime'],format='%Y-%m-%d %H:%M:%S',errors='coerce')
        #self.data=self.data.set_index('pickup_datetime')
        #delete space in col_indexs
        self.data.reset_index(drop=True,inplace=True)
        #self.data=self.data.apply(transfer_to_datetime,axis=1)
        #self.data['dropoff_datetime']=pd.to_datetime(self.data['dropoff_datetime'],format='%Y-%m-%d %H:%M:%S')
        #self.data['pickup_datetime']=self.data['pickup_datetime'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        #self.data['dropoff_datetime']=self.data['dropoff_datetime'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        #self.data=self.data[type(self.data['pickup_datetime'])==str&type(self.data['dropoff_datetime'])==str]
        #drop NAT datetime
        print(self.data.dtypes,len(self.data))
        self.clean()
        print(self.data.dtypes,len(self.data))
        #self.data.reset_index(drop=True,inplace=True)

        #self.sort_date()
        print("🤣success: data init completed in",time.time()-start," s with length", self.length())
    
    def concat_data(self,data_handler=None,fare_handler=None):
        self.data=pd.concat([data_handler,fare_handler],axis=1)
        
    def sort_date(self):
        start=time.time()
        print("🤖init:data_sort...")
        self.data.sort_values(by=['pickup_datetime'],key=lambda x: x.apply(lambda y: (y-configure.start_date).total_seconds()),inplace=True)
        print("🤣success:data_sort completed in",time.time()-start," s")
    def random_num_requests(self,num):
        #get  num of random requests
        return self.data.sample(n=num,replace=False)
    
    def to_count(self,c):
        self.count=c
    
    def next(self):
        self.count+=1
    
    def request(self):
        return self.data.iloc[self.count]
    
    def length(self):
        return len(self.data)
    
    def clean(self):
        print("🤖init:data_clean...")
        bound_index=(self.data['trip_distance']>=0)&(self.data['trip_distance']<configure.distance_seg[-1])&(self.data['trip_time_in_secs']>=0)&(self.data['trip_time_in_secs']<configure.time_seg[-1])
        self.data=self.data[bound_index]
        self.data=self.data[self.data['fare_amount']>0]
        self.data=self.data.dropna(how='any',axis='rows')
        self.data=self.data[select_within_boundingbox(self.data,configure.boundingbox)]
        self.data=remove_datapoints_from_water(self.data)

    def delete_before(self):
        self.data=self.data.drop(self.data.index[:self.count-1])
#print(d.get_request(pd.to_datetime('2013-01-01 00:20:00', format='%Y-%m-%d %H:%M:%S')))

def _init():#初始化
    global _global_dict
    try:
        _global_dict
    except:
        _global_dict = {}
def exist_key(key):
    return key in _global_dict
def set_value(key,value):
    try:
        _global_dict[key] = value
    except:
        _init()
def get_value(key,defValue=None):
    try:
        return _global_dict[key]
    except: 
        if key=='handler':
            set_value('handler',datahandler())
            return _global_dict[key]
