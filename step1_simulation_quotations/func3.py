    #a little bit more complex linear regression with distance , trip time , weekday , start time
def functrain(data_num):
    import data_vision
    import env
    import numpy as np
    import pandas as pd
    import datetime
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
    price=train_df['fare_amount']
    x=train_df[['trip_distance','trip_time_in_secs','minutes_in_the_day','monday','tuesday','wednesday','thursday','friday','saturday','sunday']]

    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split,cross_val_score

    x_train,x_test,y_train,y_test=train_test_split(x,price,test_size=0.2)
    lr=LinearRegression()
    import joblib
    lr.fit(x_train, y_train)
    joblib.dump(lr, 'funcsmodel/func3.pkl')

    y_pred = lr.predict(x_test)
    from sklearn import metrics
    MSE = metrics.mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    #MSE: 7.925403182822363
    #RMSE: 2.8152092609293473
    # 2.70933445e+00 -1.86935243e-03  8.66345539e-04 -7.83345153e-01 -7.04312242e-02 -7.79778984e-01  1.64216648e-01 -2.15206859e-02 1.57364066e+00 -8.27812596e-02 3.49
    return RMSE

