import sklearn
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import configure
data=pd.read_hdf('traindata.h5','data')
testdata=pd.read_hdf('testdata.h5','data')
x=data[['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude','minutes_in_day','weekday','trip_time_in_secs','trip_distance']].values
d2=data[['quotation'+str(i) for i in range(10)]]
y=d2[['quotation0','quotation5','quotation9']].values
# linear prediction model
result=pd.DataFrame(data=testdata[['quotation'+str(i) for i in range(10)]].values,columns=['quotation'+str(i) for i in range(10)])
result.insert(loc=result.shape[1],column='linear_min',value=0)
result.insert(loc=result.shape[1],column='linear_mid',value=0)
result.insert(loc=result.shape[1],column='linear_max',value=0)
result.insert(loc=result.shape[1],column='poly_min',value=0)
result.insert(loc=result.shape[1],column='poly_mid',value=0)
result.insert(loc=result.shape[1],column='poly_max',value=0)
linearModels=[]
for i in range(len(configure.percentiles)):
    linearModels.append(LinearRegression())
    linearModels[i].fit(x,y[:,i])

result['linear_min']=linearModels[0].predict(testdata[['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude','minutes_in_day','weekday','trip_time_in_secs','trip_distance']])
result['linear_mid']=linearModels[1].predict(testdata[['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude','minutes_in_day','weekday','trip_time_in_secs','trip_distance']])
result['linear_max']=linearModels[2].predict(testdata[['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude','minutes_in_day','weekday','trip_time_in_secs','trip_distance']])


    

#polynomial prediction model
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=2)
poly_features=poly.fit_transform(x)
poly_linearModels=[]
for i in range(len(configure.percentiles)):
    poly_linearModels.append(LinearRegression())
    poly_linearModels[i].fit(poly_features,y[:,i])

result['poly_min']=poly_linearModels[0].predict(poly.fit_transform(testdata[['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude','minutes_in_day','weekday','trip_time_in_secs','trip_distance']]))
result['poly_mid']=poly_linearModels[1].predict(poly.fit_transform(testdata[['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude','minutes_in_day','weekday','trip_time_in_secs','trip_distance']]))
result['poly_max']=poly_linearModels[2].predict(poly.fit_transform(testdata[['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude','minutes_in_day','weekday','trip_time_in_secs','trip_distance']]))


result.to_hdf('tri_result.h5','data')
