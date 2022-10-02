from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import re
import os
names=[
    'randomquo',
    'LinTS',
    'greedy',
    'Tom',
    'ucb'
]
log_dirs=['heurlogs/high_cost_2_'+names[i]+'/' for i in range(len(names))]
new_dirs=['newlogs/high_cost_2_'+names[i]+'/' for i in range(len(names))]
epochs=[('epo'+str(i)) for i in range(20)]
def save_data(num):
    for i in range(20):
        dirs=os.listdir(log_dirs[num]+epochs[i])
        for dir in dirs:
            if re.match('cum_reward',dir):
                args=event_accumulator.EventAccumulator(log_dirs[num]+epochs[i]+'/'+dir)
                event=args.Reload()
                dd=pd.DataFrame(event.scalars.Items('cum_reward'))
                dir_nums=re.findall(r'[\d.\s]+',dir)[0]
                if not os.path.exists(new_dirs[num]):
                    os.makedirs(new_dirs[num])
                if not os.path.exists(new_dirs[num]+dir_nums):
                    os.mkdir(new_dirs[num]+dir_nums)
                dd.to_csv(new_dirs[num]+dir_nums+'/'+epochs[i] +'.csv',index=False)
                pass
if __name__ == '__main__':
    for i in range(len(log_dirs)):
        save_data(i)
# args=event_accumulator.EventAccumulator('heurlogs/greedy/epo0/cum_reward_cum_rewardmutLinTS0 0.1 60 120/events.out.tfevents.1663309481.amax.1183232.2')
# event=args.Reload()
# pd=pd.DataFrame(event.scalars.Items('cum_reward'))
# pd.to_csv('temp.csv')
# pass