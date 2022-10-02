from turtle import radians
import numpy as np
import data_vision
import env
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import joblib
import sklearn
from warnings import simplefilter
import configure
simplefilter(action='ignore', category=FutureWarning)


def driver_ps(request,driver,enviroment,history=[]):
    return []
#!testing
#distance greedy method
    lmin=100000
    dri=None
    for d in driver:
        if d['online']==True:
            l=abs(d['location'][0]-request['pickup_longitude'])+abs(d['location'][1]-request['pickup_latitude'])
            if lmin>l:
                lmin=l
                dri=d
    return dri


def func1(request,driver,enviroment,history=[]):
    #func1.py
    #rmse 3.59441
    #w=[2.704,-0.002,4.319]
    print('generating func1')
    if not data_vision.exist_key('func1model'):
        data_vision.set_value('func1model',joblib.load('funcsmodel/func1.pkl'))
    a=data_vision.get_value('func1model')
    train_df=request.copy(deep=True)
    predic=a.predict(train_df[['trip_distance','trip_time_in_secs']])
    
    dri=driver_ps(request,driver,enviroment)
    return predic,dri
    


def func2(request,driver,enviroment,history=[]):
    #func2.py
    print('generating func2')
    dri=driver_ps(request,driver,enviroment)
    
    predic=request['fare_amount']+np.random.normal(0,2)
    return predic,dri

def func3(request,driver,enviroment,history=[]):
    #func3.py
    print('generating func3')
    if not data_vision.exist_key('func3model'):
        data_vision.set_value('func3model',joblib.load('funcsmodel/func3.pkl'))
    a=data_vision.get_value('func3model')
    train_df=request.copy(deep=True)
    train_df['minutes_in_the_day']=train_df['pickup_datetime'].apply(lambda x:x.hour*60+x.minute)
    train_df['monday']=train_df['pickup_datetime'].apply(lambda x:1 if x.weekday()==0 else 0)
    train_df['tuesday']=train_df['pickup_datetime'].apply(lambda x:1 if x.weekday()==1 else 0)
    train_df['wednesday']=train_df['pickup_datetime'].apply(lambda x:1 if x.weekday()==2 else 0)
    train_df['thursday']=train_df['pickup_datetime'].apply(lambda x:1 if x.weekday()==3 else 0)
    train_df['friday']=train_df['pickup_datetime'].apply(lambda x:1 if x.weekday()==4 else 0)
    train_df['saturday']=train_df['pickup_datetime'].apply(lambda x:1 if x.weekday()==5 else 0)
    train_df['sunday']=train_df['pickup_datetime'].apply(lambda x:1 if x.weekday()==6 else 0)
    x=train_df[['trip_distance','trip_time_in_secs','minutes_in_the_day','monday','tuesday','wednesday','thursday','friday','saturday','sunday']]


    #w=[2.70933445e+00,-1.86935243e-03,8.66345539e-04,-7.83345153e-01,-7.04312242e-02,-7.79778984e-01,1.64216648e-01,-2.15206859e-02,1.57364066e+00,-8.27812596e-02,3.49]
    predic=a.predict(x)
    dri=driver_ps(request,driver,enviroment)
    return predic,dri

def func4(request,driver,enviroment,history=[]):
    #func4.py
    print('generating func4')
    if not data_vision.exist_key('func4model'):
        model=joblib.load('funcsmodel/func4.pkl')
        data_vision.set_value('func4model',model)
    model=data_vision.get_value('func4model')
    
    
    def transform(data):
        data['hour'] = data['pickup_datetime'].dt.hour
        data['day'] = data['pickup_datetime'].dt.day
        data = data.drop('pickup_datetime', axis=1)
    
    train_df=request.copy(deep=True)
    transform(train_df)
    #print(train_df)
    x=train_df[['trip_distance','trip_time_in_secs','hour','day','pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']]
    predic=model.predict(x)
    dri=driver_ps(request,driver,enviroment)
    return predic,dri


def func5(request,driver,enviroment,history=[]):
    print('generating func5')
    if not data_vision.exist_key('func5model'):
        data_vision.set_value('func5model',joblib.load('funcsmodel/func5.pkl'))
    a=data_vision.get_value('func5model')
    train_df=request.copy(deep=True)
    train_df['weekday']=train_df['pickup_datetime'].dt.weekday
    train_df['minute']=train_df['pickup_datetime'].dt.minute+train_df['pickup_datetime'].dt.hour*60
    train_df=train_df[[ 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'weekday', 'minute', 'fare_amount','trip_distance','trip_time_in_secs']]
    request=train_df.drop('fare_amount',axis=1)
    
    predic=a.predict(request)
    dri=driver_ps(request,driver,enviroment)
    return predic,dri

def func6(request,driver,enviroment,history=[]):
    print('generating func6')
    train_df=request.copy(deep=True)
    train_df['minutes_in_the_day']=train_df['pickup_datetime'].apply(lambda x:x.hour*60+x.minute)
    train_df['monday']=train_df['pickup_datetime'].apply(lambda x:1 if x.weekday()==0 else 0)
    train_df['tuesday']=train_df['pickup_datetime'].apply(lambda x:1 if x.weekday()==1 else 0)
    train_df['wednesday']=train_df['pickup_datetime'].apply(lambda x:1 if x.weekday()==2 else 0)
    train_df['thursday']=train_df['pickup_datetime'].apply(lambda x:1 if x.weekday()==3 else 0)
    train_df['friday']=train_df['pickup_datetime'].apply(lambda x:1 if x.weekday()==4 else 0)
    train_df['saturday']=train_df['pickup_datetime'].apply(lambda x:1 if x.weekday()==5 else 0)
    train_df['sunday']=train_df['pickup_datetime'].apply(lambda x:1 if x.weekday()==6 else 0)
    request=train_df[['trip_distance','trip_time_in_secs','minutes_in_the_day','monday','tuesday','wednesday','thursday','friday','saturday','sunday']]

    #x=train_df[['trip_distance','trip_time_in_secs','minutes_in_the_day','monday','tuesday','wednesday','thursday','friday','saturday','sunday']]
    if not data_vision.exist_key('func6model'):
        mod=[]
        for i in range(len(configure.time_seg)-1):
            for j in range(len(configure.distance_seg)-1):
                a=joblib.load('funcsmodel/func6_time_seg'+str(i)+'_distance_seg'+str(j)+'.pkl')
                mod.append(a)
        data_vision.set_value('func6model',mod)
    #calculate time_seg and distance_seg from configure
    predic=pd.DataFrame(columns=['oriindex','predic'])
    for i in range(len(configure.time_seg)-1):
        for j in range(len(configure.distance_seg)-1):
            a=data_vision.get_value('func6model')[i*(len(configure.distance_seg)-1)+j]
            index=(request['trip_distance']>=configure.distance_seg[j])&(request['trip_distance']<configure.distance_seg[j+1])&(request['trip_time_in_secs']>=configure.time_seg[i])&(request['trip_time_in_secs']<configure.time_seg[i+1])
            x=request[index]
            pred=a.predict(x)
            index=index[index==True].index.tolist()
            p=pd.DataFrame({'oriindex':index,'predic':pred})
            predic=predic.append(p)
    predic.sort_values('oriindex',inplace=True)
    predic=list(predic['predic'])
    dri=driver_ps(request,driver,enviroment)
    return predic,dri

def func7(request,driver,enviroment,history=[]):
    print('generating func7')
    if not data_vision.exist_key('func7model'):
        data_vision.set_value('func7model',joblib.load('funcsmodel/func7.pkl'))
    train_df=request.copy(deep=True)
    train_df['minutes_in_the_day']=train_df['pickup_datetime'].apply(lambda x:x.hour*60+x.minute)
    train_df['monday']=train_df['pickup_datetime'].apply(lambda x:1 if x.weekday()==0 else 0)
    train_df['tuesday']=train_df['pickup_datetime'].apply(lambda x:1 if x.weekday()==1 else 0)
    train_df['wednesday']=train_df['pickup_datetime'].apply(lambda x:1 if x.weekday()==2 else 0)
    train_df['thursday']=train_df['pickup_datetime'].apply(lambda x:1 if x.weekday()==3 else 0)
    train_df['friday']=train_df['pickup_datetime'].apply(lambda x:1 if x.weekday()==4 else 0)
    train_df['saturday']=train_df['pickup_datetime'].apply(lambda x:1 if x.weekday()==5 else 0)
    train_df['sunday']=train_df['pickup_datetime'].apply(lambda x:1 if x.weekday()==6 else 0)
    #x=train_df[['trip_distance','trip_time_in_secs','minutes_in_the_day','monday','tuesday','wednesday','thursday','friday','saturday','sunday']]

    import cudf 
    train_df=cudf.from_pandas(train_df)
    x=train_df[['trip_distance','trip_time_in_secs','minutes_in_the_day','monday','tuesday','wednesday','thursday','friday','saturday','sunday']]
    a=data_vision.get_value('func7model')
    predic=a.predict(x)
    dri=driver_ps(request,driver,enviroment)
    predic=predic.to_pandas()
    return predic,dri


def func8(request,driver,enviroment,history=[]):
    print('generating func8')
    if not data_vision.exist_key('func8model'):
        data_vision.set_value('func8model',joblib.load('funcsmodel/func8.pkl'))
    train_df=request.copy(deep=True)
    train_df['weekday']=train_df['pickup_datetime'].dt.weekday
    train_df['minute']=train_df['pickup_datetime'].dt.minute+train_df['pickup_datetime'].dt.hour*60
    train_df=train_df[[ 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'weekday', 'minute','trip_distance','trip_time_in_secs']]
    a=data_vision.get_value('func8model')
    predic=a.predict(train_df)
    dri=driver_ps(request,driver,enviroment)
    return predic,dri


def func9(request,driver,enviroment,history=[]):
    print('generating func9')
    if not data_vision.exist_key('func9model'):
        data_vision.set_value('func9model',joblib.load('funcsmodel/func9.pkl'))
    train_df=request.copy(deep=True)
    train_df['weekday']=train_df['pickup_datetime'].dt.weekday
    train_df['minute']=train_df['pickup_datetime'].dt.minute+train_df['pickup_datetime'].dt.hour*60
    train_df=train_df[[ 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'weekday', 'minute','trip_distance','trip_time_in_secs']]
    a=data_vision.get_value('func9model')
    predic=a.predict(train_df)
    dri=driver_ps(request,driver,enviroment)
    return predic,dri


    
#test
import time
import numpy as np
funcs=[func1,func2,func3,func4,func5,func6,func7,func8,func9]
data_vision._init()
from sklearn.metrics import mean_absolute_percentage_error
for i in range(1,13):
    mape=lambda x,y:np.sqrt(mean_absolute_percentage_error(x,y))
    data_vision.set_value('handler',data_vision.datahandler(i))
    try:
        for j,func in enumerate(funcs):
            requests=data_vision.get_value('handler').data
            start=time.time()
            data_vision.get_value('handler').data['func'+str(j+1)]=func(requests,[],[],history=[])[0]
            print('file'+str(i)+'func'+str(j+1)+' finished',time.time()-start)
            print(mape(requests['fare_amount'],requests['func'+str(j+1)]))
    except:
        print('file'+str(i)+'error')
        continue
    print('start saving file:'+str(i+1))
    data_vision.get_value('handler').data.to_hdf('data/'+str(i)+'.h5','data')
        #print('error in file:'+str(i+1))


    

