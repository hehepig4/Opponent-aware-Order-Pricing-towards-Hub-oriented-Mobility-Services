def functrain(data_num):
    #segmant linear regression with distance and trip time

    import data_vision
    import configure
    d=data_vision.get_value('handler')
    train_df=d.random_num_requests(data_num)
    train_df['minutes_in_the_day']=train_df['pickup_datetime'].apply(lambda x:x.hour*60+x.minute)
    train_df['monday']=train_df['pickup_datetime'].apply(lambda x:1 if x.weekday()==0 else 0)
    train_df['tuesday']=train_df['pickup_datetime'].apply(lambda x:1 if x.weekday()==1 else 0)
    train_df['wednesday']=train_df['pickup_datetime'].apply(lambda x:1 if x.weekday()==2 else 0)
    train_df['thursday']=train_df['pickup_datetime'].apply(lambda x:1 if x.weekday()==3 else 0)
    train_df['friday']=train_df['pickup_datetime'].apply(lambda x:1 if x.weekday()==4 else 0)
    train_df['saturday']=train_df['pickup_datetime'].apply(lambda x:1 if x.weekday()==5 else 0)
    train_df['sunday']=train_df['pickup_datetime'].apply(lambda x:1 if x.weekday()==6 else 0)
    train_df=train_df[['fare_amount','trip_distance','trip_time_in_secs','minutes_in_the_day','monday','tuesday','wednesday','thursday','friday','saturday','sunday']]
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import numpy as np
    from matplotlib import pyplot as plt
    import joblib

    #divide the data into 16 parts base on configura.time_seg and configure.distance_seg
    train_df,test_df=train_test_split(train_df,test_size=0.2)
    x_train_list=[]
    y_train_list=[]
    for i in range(len(configure.time_seg)-1):
        for j in range(len(configure.distance_seg)-1):
            x_train=train_df[(train_df['trip_time_in_secs']>=configure.time_seg[i])&(train_df['trip_time_in_secs']<configure.time_seg[i+1])&(train_df['trip_distance']>=configure.distance_seg[j])&(train_df['trip_distance']<configure.distance_seg[j+1])]
            y_train=x_train['fare_amount']
            x_train=x_train.drop(['fare_amount'],axis=1)
            x_train_list.append(x_train)
            y_train_list.append(y_train)
    #train the model

    reg=[]
    for i in range(len(x_train_list)):
        print('iter:',i)
        reg.append(LinearRegression())
        reg[i].fit(x_train_list[i],y_train_list[i])
        joblib.dump(reg[i], 'funcsmodel/func6_time_seg'+str(int(i/(len(configure.distance_seg)-1)))+'_distance_seg'+str(i % (len(configure.distance_seg)-1))+'.pkl')

    #test the model
    x_test_list=[]
    y_test_list=[]
    data_num_list=[]
    for i in range(len(configure.time_seg)-1):
        for j in range(len(configure.distance_seg)-1):
            x_test=test_df[(test_df['trip_time_in_secs']>=configure.time_seg[i])&(test_df['trip_time_in_secs']<configure.time_seg[i+1])&(test_df['trip_distance']>=configure.distance_seg[j])&(test_df['trip_distance']<configure.distance_seg[j+1])]
            y_test=x_test['fare_amount']
            x_test=x_test.drop(['fare_amount'],axis=1)
            x_test_list.append(x_test)
            y_test_list.append(y_test)
            data_num_list.append(len(x_test))
    num_total=sum(data_num_list)
    for i in range(len(data_num_list)):
        data_num_list[i]=data_num_list[i]/num_total
    rmse_list=[]
    for i in range(len(x_test_list)):
        rmse_list.append(np.sqrt(mean_squared_error(y_test_list[i],reg[i].predict(x_test_list[i]))))

    return np.dot(rmse_list,data_num_list)
    
            