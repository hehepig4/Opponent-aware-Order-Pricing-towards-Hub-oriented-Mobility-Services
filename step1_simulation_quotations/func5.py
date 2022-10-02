

def functrain(data_num):
    #KNN
    import joblib
    from data_vision import datahandler
    import data_vision
    d=data_vision.get_value('handler')


    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import numpy as np
    import matplotlib.pyplot as plt

    train_df=d.random_num_requests(data_num)

    train_df['weekday']=train_df['pickup_datetime'].dt.weekday
    train_df['minute']=train_df['pickup_datetime'].dt.minute+train_df['pickup_datetime'].dt.hour*60
    train_df=train_df[[ 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'weekday', 'minute', 'fare_amount','trip_distance','trip_time_in_secs']]
    y=train_df['fare_amount']
    x=train_df.drop('fare_amount',axis=1)

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

    knn=KNeighborsRegressor(n_neighbors=5)
    knn.fit(x_train,y_train)
    joblib.dump(knn, 'funcsmodel/func5.pkl')

    rmse=np.sqrt(mean_squared_error(y_test,knn.predict(x_test)))
    return rmse
