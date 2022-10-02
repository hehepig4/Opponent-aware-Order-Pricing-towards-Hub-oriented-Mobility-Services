def functrain(data_num):
    #simple regression with distance and trip time
    import data_vision
    import pandas as pd
    import numpy as np
    import joblib
    import cudf
    d=data_vision.get_value('handler')
    train_df=d.random_num_requests(data_num)
    
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    tag=train_df['fare_amount']
    train_df=train_df[['trip_distance','trip_time_in_secs']]


    
    X_train, X_test, y_train, y_test = train_test_split(train_df, tag, test_size=0.5, random_state=1)

    lr = LinearRegression()

    lr.fit(X_train, y_train)
    joblib.dump(lr, 'funcsmodel/func1.pkl')

    y_pred = lr.predict(X_test)
    from sklearn.metrics import mean_squared_error
    MSE = mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(MSE)
    return RMSE

