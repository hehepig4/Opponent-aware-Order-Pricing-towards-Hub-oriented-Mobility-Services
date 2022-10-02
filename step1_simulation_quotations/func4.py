    #bayesianridge regression
def functrain(data_num):
    from sklearn.linear_model import BayesianRidge
    import data_vision
    import env
    import numpy as np
    import pandas as pd
    import datetime
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    d=data_vision.get_value('handler')


    def transform(data):
        data['hour'] = data['pickup_datetime'].dt.hour
        data['day'] = data['pickup_datetime'].dt.day
        data = data.drop('pickup_datetime', axis=1)
        
    train_df=d.random_num_requests(data_num)
    transform(train_df)
    train_df=train_df[['fare_amount','trip_distance','trip_time_in_secs','hour','day','pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']]
    price=train_df['fare_amount']
    x=train_df.drop('fare_amount',axis=1)
    x_train,x_test,y_train,y_test=train_test_split(x,price,test_size=0.2)
    model=BayesianRidge()
    model.fit(x_train,y_train)
    import joblib
    joblib.dump(model, 'funcsmodel/func4.pkl')
    y_pred = model.predict(x_test)
    return np.sqrt(mean_squared_error(y_test, y_pred))
"""
    dtrain=xgb.DMatrix(x_train,label=y_train)
    dtest=xgb.DMatrix(x_test)


    def xgb_evaluate(max_depth, gamma, colsample_bytree):
        params = {'eval_metric': 'rmse',
                'max_depth': int(max_depth),
                'subsample': 0.8,
                'eta': 0.1,
                'gamma': gamma,
                'colsample_bytree': colsample_bytree}
        # Used around 1000 boosting rounds in the full model
        cv_result = xgb.cv(params, dtrain, num_boost_round=100, nfold=3)    
        
        # Bayesian optimization only knows how to maximize, not minimize, so return the negative RMSE
        return -1.0 * cv_result['test-rmse-mean'].iloc[-1]

    xgb_bo = BayesianOptimization(xgb_evaluate, {'max_depth': (3, 7), 
                                                'gamma': (0, 1),
                                                'colsample_bytree': (0.3, 0.9)})
    from sklearn.metrics import mean_squared_error
    xgb_bo.maximize(init_points=3, n_iter=5, acq='ei')

    params = xgb_bo.max['params']
    params['max_depth'] = int(params['max_depth'])
    model=xgb.train(params,dtrain,num_boost_round=250)
    model.save_model('funcsmodel\\func4.pkl')

    y_pred = model.predict(dtest)
    return np.sqrt(mean_squared_error(y_test, y_pred))


    import matplotlib.pyplot as plt
    fscores = pd.DataFrame({'X': list(model.get_fscore().keys()), 'Y': list(model.get_fscore().values())})
    fscores.sort_values(by='Y').plot.bar(x='X')
    #rmse:2.4073231

"""
