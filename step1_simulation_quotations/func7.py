    #SVM regression
def functrain(data_num):
    from cuml.svm import SVR
    import data_vision
    import pandas as pd  
    rbf=SVR(kernel='rbf',C=1e3,gamma=0.1,verbose=1)
    d=data_vision.get_value('handler')
    train_df=d.random_num_requests(data_num)

    train_df=pd.concat([train_df['fare_amount'],train_df['trip_time_in_secs'],train_df['trip_distance'],train_df['pickup_datetime']],axis=1)
    train_df['minutes_in_the_day']=train_df['pickup_datetime'].apply(lambda x:x.hour*60+x.minute)
    train_df['monday']=train_df['pickup_datetime'].apply(lambda x:1 if x.weekday()==0 else 0)
    train_df['tuesday']=train_df['pickup_datetime'].apply(lambda x:1 if x.weekday()==1 else 0)
    train_df['wednesday']=train_df['pickup_datetime'].apply(lambda x:1 if x.weekday()==2 else 0)
    train_df['thursday']=train_df['pickup_datetime'].apply(lambda x:1 if x.weekday()==3 else 0)
    train_df['friday']=train_df['pickup_datetime'].apply(lambda x:1 if x.weekday()==4 else 0)
    train_df['saturday']=train_df['pickup_datetime'].apply(lambda x:1 if x.weekday()==5 else 0)
    train_df['sunday']=train_df['pickup_datetime'].apply(lambda x:1 if x.weekday()==6 else 0)
    
    import cudf
    train_df=cudf.from_pandas(train_df)
    price=train_df['fare_amount']
    x=train_df[['trip_distance','trip_time_in_secs','minutes_in_the_day','monday','tuesday','wednesday','thursday','friday','saturday','sunday']]

    from cuml.model_selection import train_test_split

    x_train,x_test,y_train,y_test=train_test_split(x,price,test_size=0.2)

    rbf.fit(x_train, y_train)

    import joblib
    joblib.dump(rbf, 'funcsmodel/func7.pkl')
    y_pred = rbf.predict(x_test)
    from sklearn import metrics
    import numpy as np
    y_test=y_test.to_numpy()
    y_pred=y_pred.to_numpy()
    return np.sqrt(metrics.mean_squared_error(y_test, y_pred))

