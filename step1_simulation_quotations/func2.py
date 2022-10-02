#real fare with Gauss noise
def functrain(data_num):
    import data_vision
    import pandas as pd
    import numpy as np
    d=data_vision.get_value('handler')
    train_df=d.random_num_requests(data_num)


    from cuml.linear_model import LinearRegression
    from cuml.model_selection import train_test_split
    tag=train_df['fare_amount']
    pre=train_df['fare_amount'].apply(lambda x:x+np.random.normal(0,2))

    from sklearn import metrics
    MSE = metrics.mean_squared_error(tag,pre)
    RMSE = np.sqrt(metrics.mean_squared_error(tag,pre))
    return RMSE
