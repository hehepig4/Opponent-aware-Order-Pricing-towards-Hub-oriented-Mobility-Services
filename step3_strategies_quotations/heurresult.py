import os
import pandas as pd
names=[
    'randomquo',
    'LinTS',
    'greedy',
    'Tom',
    'ucb'
]
dirs=[
    'newlogs/high_cost_2_'+names[i]+'/' for i in range(len(names))
]
for i in range(len(dirs)):
    for dir in os.listdir(dirs[i]):
        average_handle=0
        max_handle=(-1,0)
        for epo in range(20):
            data=pd.read_csv(dirs[i]+dir+'/epo'+str(epo)+'.csv')
            ending_reward=data['value'].iloc[-1]
            if ending_reward>max_handle[1]:
                max_handle=(epo,ending_reward)
            average_handle+=ending_reward
        average_handle/=20
        print('name: '+names[i]+' dir: '+dir+' average: '+str(average_handle)+' max: '+str(max_handle))