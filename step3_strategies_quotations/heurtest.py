import heuristic_agent
import configure
import sac
import utils
import enviroment
import torch
import numpy as np
import pandas as pd
import datahandler
from tqdm import tqdm
from rich import console,progress
import gym
import envpool
import ray
import time
ray.init()


data=pd.read_hdf('traindata.h5')
dataset=datahandler.requesthandler(data)
datahandler.set_value('data',data)
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
# alos=[1,3,5,10]
# per=[0.05,0.1,0.2]
# indexing=[(60,120),(1000,1060)]
alos=[0,3,5]
per=[0.05,0.1,0.2]
indexing=[(500,560)]
def eval(epochs,cost_rate=0.237,init_update=0,load=-1,device='cpu'):
    
    writter=SummaryWriter(log_dir='heurlogs/high_cost_2_randomquo/epo'+str(epochs))
    data_=pd.read_hdf('test_data/reqdata.h5')
    qhandle=torch.load('test_data/Qhandle.pt')
    timerec=torch.load('test_data/timerec.pt')
    Zdata=torch.load('test_data/Zdata.pt')
    history_data=pd.read_hdf('test_data/new_traindata.h5')
    #zdata=
    #Qdata=
    data_id=[]
    data_id.append(ray.put(data_))
    data_id.append(ray.put(qhandle))
    data_id.append(ray.put(timerec))
    data_id.append(ray.put(Zdata))
    
    update_times=init_update*5
    progressId={}
    arg_list=[]
    s_e_handle=[]
    arghandle=[]
    for i in range(len(alos)):
        for j in range(len(per)):
            for (start,end) in indexing:
                arghandle.append('randomquo'+str(alos[i])+' '+str(per[j])+' '+str(start)+' '+str(end))
                arg_list.append((alos[i],per[j],start,end,data_id,cost_rate,device))
    
    env=gym.make('CarPool-v1',device=device,args_list=arg_list,history_data=history_data,agent_nums=np.array([1],dtype=np.float32))
    obs=env.reset()
    finished=False
    step=0
    agents=heuristic_agent.multiAgents([
        #heuristic_agent.multiArmBanditAgent.remote(arms=9) for _ in range(len(arg_list))
        #heuristic_agent.mutgreedAgent.remote() for _ in range(len(arg_list))
        #heuristic_agent.estimatemutiArmBanditAgent.remote(arms=9) for _ in range(len(arg_list))
        #heuristic_agent.multiQuoMultiArmBanditAgent.remote(arms=9) for _ in range(len(arg_list))
        #heuristic_agent.multiTomsArmBanditAgent.remote(arms=9) for _ in range(len(arg_list))
        #heuristic_agent.MulLinTSMultiArmbanditAgent.remote(arms=9,dimension=5) for _ in range(len(arg_list))
        heuristic_agent.LinTSMultiArmbanditAgent.remote(arms=9,dimension=5) for _ in range(len(arg_list))
        #heuristic_agent.randomAgent.remote() for _ in range(len(arg_list))
    ],arghandle
    )
    info={'driver_nums':np.zeros((len(arg_list),11),dtype=np.float32)}
    with tqdm() as pbar:
        while not finished:
            
            step+=1
            #t=time.perf_counter()
            act=agents.acts(*obs,info)
            #print(time.perf_counter()-t)
            obs_next,rewards,done,info=env.step((act,None))
            pbar.total=info['all'][0]
            pbar.n=info['done'][0]
            pbar.update(0)
            if any(done):
                finished=True
            Tr=[]
            for i in range(len(act)):
                o_=[]
                no_=[]
                for j in range(3):
                    o_.append(obs[j][i])
                    no_.append(obs_next[j][i])
                Tr.append(utils.constructTran(o_,act[i],no_,rewards[i],done[i]))
            agents.updating(Tr)
            obs=obs_next
            logging(info,step,writter,arghandle,obs[0][0],obs[1][0],act)
        
        
            #env=gym.make('CarPool-v0',driver_nums=dri_nums,cost_rate=cost_rate,device=device,envid=0) 
            #env=gym.vector.make('CarPool-v0',num_envs=8,asynchronous=False,dri_nums=dri_nums,cost_rate=cost_rate,device=device,envid=0)
            
                         
def logging(info,step,writter,arglist,Q,Q_dist,act,numagents=len(configure.cost_rate)):
    cum_rewards=info['cum_reward']
    cum_reward_dict={}
    for i in range(len(cum_rewards)):
        cum_reward_dict['cum_reward'+arglist[i]]=cum_rewards[i]
    writter.add_scalars('cum_reward',cum_reward_dict,step)
    drivers=info['driver_nums']
    driver_dict={}
    for i in range(len(drivers)):
        sumdri=np.sum(drivers[i])
        driver_dict['driver_per'+arglist[i]]=0
        for j in range(numagents):
            driver_dict['driver_per'+arglist[i]]+=drivers[i][j]/sumdri
    writter.add_scalars('driver_per',driver_dict,step)
    getted_req=info['getted_req']
    getted_req_dict={}
    reward_per_req_dict={}
    for i in range(len(getted_req)):
        getted_req_dict['getted_req'+arglist[i]]=getted_req[i]/step
        reward_per_req_dict['reward_per_req'+arglist[i]]=0 if getted_req[i]==0 else cum_rewards[i]/getted_req[i]
    writter.add_scalars('getted_req',getted_req_dict,step)
    writter.add_scalars('reward_per_req',reward_per_req_dict,step)
    Q_dist_dict={}
    for i in range(len(configure.Qquan)):
        Q_dist_dict.update({'Q_dist'+str(configure.Qquan[i]):Q_dist[0][i].item()})
    Q_dist_dict.update({'Q_value':Q.item()})
    writter.add_scalars('Q_value',Q_dist_dict,step)
    act_dist={}
    for i in range(len(getted_req)):
        for j in range(numagents):
            act_dist.update({'act'+arglist[i]+'_'+str(j):act[i][j].item()})
    writter.add_scalars('act',act_dist,step)
if __name__=='__main__':
    for i in range(20):
        eval(i)