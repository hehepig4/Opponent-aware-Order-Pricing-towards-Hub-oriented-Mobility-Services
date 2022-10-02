#merge sort  data1-12 into data.h5 with datetime
import time
import pandas as pd
import numpy as np
start=time.time()
handle_list=[]
print('start')
sum_length=0
for i in range(1,13):
    print('reading data',i)
    try:
        temp=pd.read_hdf('data/'+str(i)+'.h5')
        handle_list.append(temp)
        sum_length+=len(temp)
    except:
        print('data',i,'not exist')
print('finish reading data in',time.time()-start,'s')
start=time.time()
print('start merging data')

#merge sort
import configure
res=pd.concat(handle_list)
del handle_list

res=res[['pickup_datetime', 'dropoff_datetime', 
       'trip_time_in_secs', 'trip_distance', 
       'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 
       'fare_amount',  'func1', 'func2', 'func3',  'func4',  'func5', 'func6', 'func7', 'func8', 'func9']]
print(len(res))
for i in range(9):
    res=res[res['func'+str(i+1)]>0]
print(len(res))
res.sort_values(by=['pickup_datetime'],key=lambda x:  x.apply(lambda y: (y-configure.start_date).total_seconds()),inplace=True)
print('finish merging data in',time.time()-start,'s')
start=time.time()
print('start saving data')
res.to_hdf('data/data.h5','data')
