

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from collections import namedtuple,deque
from gym.envs.registration import register
transistion=namedtuple('transistion',['state','action','next_state','reward','done'])
import ray
def constructTran(state,action,next_state,reward,done):
    return transistion(state,action,next_state,reward,done)
#for continuous
'''
    state: [batch,feature_input]
    action: [batch,1]
    next_state: [batch,feature_input]
    reward: [batch,1]
    done: [batch,1]
'''
#for discrete
'''
    state: [batch,feature_input]
    action: [batch,action_space]
    next_state: [batch,feature_input]
    reward: [batch,1]
    done:[batch,1]
'''
class replaybuffer():
    def __init__(self,capacity,batch_size,alpha=0.6,beta=0.4,beta_increment_per_sampling=0.001):
        self.e=0.001
        self.capacity=capacity
        self.batch_size=batch_size
        self.alpha=alpha
        self.beta=beta
        self.beta_increment_per_sampling=beta_increment_per_sampling
        class Sum_Tree():
            def __init__(self,capacity):
                self.tree=np.zeros(2*capacity-1)
                self.data=np.zeros(capacity,dtype=object)
                self.entries=0
                self.write=0
                self.capacity=capacity
            def _propagate(self, idx, change):
                parent = (idx - 1) // 2
                self.tree[parent] += change
                if parent != 0:
                    self._propagate(parent, change)
            def _retrieve(self, idx, s):
                left = 2 * idx + 1
                right = left + 1
                if left >= len(self.tree):
                    return idx
                if s <= self.tree[left]:
                    return self._retrieve(left, s)
                else:
                    return self._retrieve(right, s - self.tree[left])
            def _update(self,idx,p):
                change=p-self.tree[idx]
                self.tree[idx]=p
                self._propagate(idx,change)
                
            def _add(self, p,data):
                idx=self.write+self.capacity-1
                self.data[self.write]=data
                self._update(idx,p)
                self.write+=1
                if self.write>=self.capacity:
                    self.write=0
                if self.entries<self.capacity:
                    self.entries+=1
            def _get(self,s):
                idx=self._retrieve(0,s)
                dataidx=idx-self.capacity+1
                return (idx,self.tree[idx],self.data[dataidx])
            def total(self):
                return self.tree[0]
        self.tree=Sum_Tree(capacity)
    def _get_priority(self,error):
        return (np.abs(error)+self.e)**self.alpha
    def add(self,tran,error):
        p=self._get_priority(error)
        self.tree._add(p,tran)
    def sample(self,batchs=0):
        if batchs==0:
            batchs=self.batch_size
        sampled=[]
        idxs=[]
        pris=[]
        segment=self.tree.total()/batchs
        
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        for _ in range(batchs):
            a=segment*_
            b=segment*(_+1)
            s=np.random.uniform(a,b)
            (idx,p,data)=self.tree._get(s)
            idxs.append(idx)
            pris.append(torch.tensor([p],dtype=float))
            sampled.append(data)
        mats=torch.concat([tran.state[0] for tran in sampled])
        feas=torch.concat([tran.state[1] for tran in sampled])
        dris=torch.concat([tran.state[2] for tran in sampled])
        states=[mats,feas,dris]
        
        acts=torch.concat([tran.action for tran in sampled])
        
        mats=torch.concat([tran.next_state[0] for tran in sampled])
        feas=torch.concat([tran.next_state[1] for tran in sampled])
        dris=torch.concat([tran.next_state[2] for tran in sampled])
        next_ss=[mats,feas,dris]
        
        rews=torch.concat([tran.reward for tran in sampled])
        done=torch.concat([tran.done.float() for tran in sampled])
        pris=torch.stack(pris,dim=0).float()
        pris=pris/self.tree.total()
        weights=torch.pow(self.tree.entries*pris,-self.beta)
        weights=weights/torch.max(weights,dim=0,keepdim=True)[0]
        return [states,acts,next_ss,rews,done,weights]
    # def sample(self,batchs=0):
    #     if batchs==0:
    #         batchs=self.batch_size
    #     sampled=[]
    #     while len(sampled)<batchs and len(self.memory)>0:
    #         sampled.append(self.memory.pop(np.random.randint(0,len(self.memory))))
    #     mats=torch.concat([tran.state[0] for tran in sampled])
    #     feas=torch.concat([tran.state[1] for tran in sampled])
    #     dris=torch.concat([tran.state[2] for tran in sampled])
    #     states=[mats,feas,dris]
        
    #     acts=torch.concat([tran.action for tran in sampled])
        
    #     mats=torch.concat([tran.next_state[0] for tran in sampled])
    #     feas=torch.concat([tran.next_state[1] for tran in sampled])
    #     dris=torch.concat([tran.next_state[2] for tran in sampled])
    #     next_ss=[mats,feas,dris]
        
    #     rews=torch.concat([tran.reward for tran in sampled])
    #     done=torch.concat([tran.done.float() for tran in sampled])
    #     return [states,acts,next_ss,rews,done]

import gym
import configure
import datahandler
import numpy as np
import pandas as pd
@ray.remote(num_cpus=2)
class envir(gym.core.Env):
    def init_dri(self,height,width,driver_num,sampled_min_minutes,sampled_max_minutes,default_reward=-1):
        inits=self.data_key[['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude','minutes_in_day']].copy(deep=True)
        idexs=(inits['minutes_in_day']>=sampled_min_minutes)&(inits['minutes_in_day']<=sampled_max_minutes)
        inits=inits[idexs][:].reset_index(drop=True)
        inits=inits[['pickup_latitude','pickup_longitude']].values
        assert inits.shape[0]>=driver_num
        idx=np.random.choice(a=range(inits.shape[0]),size=driver_num,replace=False)
        matrix=np.zeros((height,width),dtype=int)
        for id in idx:
            start_block_la=int(np.round((inits[id,0]-configure.min_la)/configure.step_dis))
            start_block_lo=int(np.round((inits[id,1]-configure.min_lo)/configure.step_dis))
            matrix[start_block_la,start_block_lo]+=1
        return torch.from_numpy(matrix).float()
    def get_order(self,q,quos,global_dri,init_block,ending_block,running_dict,expEndTime,cost,height,width):
        alo=configure.alo
        q_init=q
        quos_init=quos
        quos=quos.clone().numpy()
        q=q.clone().numpy()
        quos=np.random.permutation(quos.T).T
        getOrder=-1
        quotations=np.concatenate([q,quos],axis=0)
        selected=np.argpartition(quotations, alo)[:alo]
        def random_select(selected,selected_num):
            return np.random.choice(selected,p=selected_num/selected_num.sum(),size=1)[0]
        dri_block=init_block
        dist=0
        find=False
        prob_vec=np.zeros_like(selected,dtype=float)
        searched=[]
        while not find:
            searched=[]
            for f in range(dist):
                o=dist-f
                for _t in iter([(o,f),(o,-f),(-o,f),(-o,-f)]):
                    (i,j)=_t
                    dri_block=(init_block[0]+i,init_block[1]+j)
                    if dri_block[0]<0 or dri_block[0]>=height or dri_block[1]<0 or dri_block[1]>=width:
                        continue
                    if dri_block not in searched:
                        searched.append(dri_block)
                        prob_vec+=np.array([global_dri[s][dri_block[0],dri_block[1]] for s in selected])
            if np.sum(prob_vec)>0:
                find=True
            if find:
                getOrder=random_select(selected=selected,selected_num=prob_vec)
                break
            dist+=1
        for s in searched:
            if global_dri[getOrder][s[0],s[1]]>0:
                global_dri[getOrder][s[0],s[1]]-=1
                break
        if expEndTime in running_dict.keys():
            running_dict[expEndTime].append((ending_block[0],getOrder))
        else:
            running_dict[expEndTime]=[(ending_block[0],getOrder)]
        if getOrder==0:
            return q_init-cost,True
        else:
            if 0 in selected:
                return -(q_init-cost)*(torch.sum(global_dri[0])/torch.sum(global_dri[1:])),False
            else: 
                return torch.tensor([[0.]],dtype=torch.float32),False
        
    def act_to_quo(self,quos,act):
        less=act<0
        bigger=act>=0
        q=less*((quos[:,1]-quos[:,0])*act+quos[:,1])
        q+=bigger*((quos[:,-1]-quos[:,0])*act+quos[:,0])
        return q

    def iterhandler(self):
        for mats,feas,quos,preds,cost in self.reqdataloader:
            mats=mats   
            feas=torch.concat([feas,preds],dim=1)
            quos=quos[:,:5]
            preds=preds
            cost=cost*self.cost_rate
            yield (mats,feas,quos,preds,cost,self.global_dri[0].unsqueeze(0).unsqueeze(0))
        return None
    def getnext(self):
        try:
            nex=next(self._dhand)
        except:
            nex=None
        return nex
    def __init__(self,driver_nums,cost_rate,device,envid,data_key):
        self.action_space=gym.Space()
        self.observation_space=gym.Space()
        self.driver_nums=driver_nums
        self.cost_rate=cost_rate
        self.device=device
        self.envid=envid
        self.cum_reward=0
        self.hour_reward=0
        self.stepnums=0
        self.data_key=data_key
        self.last_obs=None
        self.index=0
        self.getted_req=0
    def reset(self):
        try:
            self._dhand=self.iterhandler()
            self.index=0
            self.getted_req=0
            data=self.data_key.copy(deep=True)
            begin=self.envid*60
            start_index=(data['minutes_in_day']<=(begin+1)) & (data['minutes_in_day']>=(begin-1))
            start_index=data[start_index].index.values
            start=np.random.randint(low=0,high=start_index.shape[0])
            while start!=0 and start_index[start-1]==start_index[start]-1 :
                start-=1
            start=start_index[start]
            end_index=data.iloc[start:,:]
            end_index=end_index[end_index['minutes_in_day']>=(begin+60)].index.values[0]
            data=pd.DataFrame(data.iloc[start:end_index+1,:],copy=True)
            data.reset_index(drop=True,inplace=True)
            self.reqdataloader=datahandler.requesthandler(data)
            self.reqdataloader=torch.utils.data.DataLoader(self.reqdataloader,batch_size=1,shuffle=False,num_workers=0)
            
            self.running_dri_dict={}
            global_dri=[self.init_dri(configure.input_size[0],configure.input_size[1],dris,begin-30,begin) for dris in self.driver_nums]
            self.global_dri=torch.stack(global_dri,dim=0)
            self.last_minuts=self.envid*60-1
            self.cum_reward==torch.tensor([[0.]],dtype=torch.float32)
            self.hour_reward==torch.tensor([[0.]],dtype=torch.float32)
            self.stepnums=0
            next=self.getnext()
            self.last_obs=next
            return next
        except:
            return self.reset()
    def get_driver_nums(self):
        return [torch.sum(self.global_dri[i]).item() for i in range(self.global_dri.shape[0])]
    def step(self,act_obs):
        act,obs=act_obs
        obs=self.last_obs
        self.stepnums+=1
        done=False
        mats,feas,quos,preds,cost,dri=obs
        for key in list(self.running_dri_dict.keys()):
            if key<=self.last_minuts:
                for dria in self.running_dri_dict[key]:
                    ending_block,getOrder=dria
                    self.global_dri[getOrder][ending_block[0],ending_block[1]]+=1
                del self.running_dri_dict[key]
        q=self.act_to_quo(quos=preds,act=act)
        lamb,get=self.get_order(q[0],quos[0],self.global_dri,torch.nonzero(mats[0,0,:,:])[0].tolist(),torch.nonzero(mats[0,1,:,:]).tolist(),running_dict=self.running_dri_dict,expEndTime=int(feas[0,2].item()//60+self.last_minuts+0.99),cost=cost,height=configure.input_size[0],width=configure.input_size[1])
        rewards=lamb[0]
        if get:
            self.getted_req+=1
        if self.stepnums==len(self.reqdataloader):
            done=True
        self.last_minuts=int(feas[0,1].item())
        obs_next=self.getnext()
        self.cum_reward+=rewards
        self.hour_reward+=rewards
        self.last_obs=obs_next
        if obs_next==None:
            done=True
        return [obs_next,rewards,torch.tensor([done],dtype=bool),{'cum_reward':self.cum_reward,'hour_reward':self.hour_reward,'done':self.stepnums,'all':len(self.reqdataloader),'driver_nums':self.get_driver_nums(),'minutes':self.last_minuts,'running_dri':len(self.running_dri_dict),'getted_req':self.getted_req}]
class CarPoolEnv(gym.Env):
    def __init__(self,args_list,device):
        self.device=device
        self.handle=[]
        for args in args_list:
            self.handle.append(envir.remote(*args))
        self.done_list=[False for _ in range(len(self.handle))]
        
        
    def obssToTensor(self,obss):
        element_count=len(obss[0])
        gatherList=[[] for i in range(element_count)]
        new=[]
        for i in range(element_count):
            gatherList[i].extend([obss[j][i] for j in range(len(obss))])
        for ele in gatherList:
            new.append(torch.concat(ele,dim=0))
        return new
    def reset(self):
        obss=[]
        for env in self.handle:
            obss.append(env.reset.remote())
        obs2=[]
        obs2=ray.get(obss)
        obss=self.obssToTensor(obs2)
        return obss

        
    def step(self,_):
        act,obs=_
        handle=[]
        res=[]
        for i in range(len(self.handle)):
            handle.append(self.handle[i].step.remote([act[i].unsqueeze(0),obs]))
        res=ray.get(handle)
        new_obs=[]
        rew=[]
        done=[]
        info=[]
        for i in range(len(self.handle)):
            new_obs.append(res[i][0])
            rew.append(res[i][1])
            done.append(res[i][2])
            info.append(res[i][3])
        for i in range(len(self.handle)):
            if done[i]:
                new_obs[i]=ray.get(self.handle[i].reset.remote())
                info[i]={'cum_reward':torch.tensor([[0.]],dtype=torch.float32),'hour_reward':torch.tensor([[0.]],dtype=torch.float32),'done':0,'all':0,'driver_nums':ray.get(self.handle[i].get_driver_nums.remote()),'minutes':0,'running_dri':0,'getted_req':0}
        new_obs=self.obssToTensor(new_obs)
        new_info={
            'cum_reward':np.array([info[i]['cum_reward'].item() for i in range(len(self.handle))]),
            'hour_reward':np.array([info[i]['hour_reward'].item() for i in range(len(self.handle))]),
            'done':np.array([info[i]['done'] for i in range(len(self.handle))]),
            'all':np.array([info[i]['all'] for i in range(len(self.handle))]),
            'driver_nums':np.array([info[i]['driver_nums'] for i in range(len(self.handle))]),
            'minutes':np.array([info[i]['minutes'] for i in range(len(self.handle))]),
            'running_dri':np.array([info[i]['running_dri'] for i in range(len(self.handle))]),
            'getted_req':np.array([info[i]['getted_req'] for i in range(len(self.handle))])
        }
        return new_obs,torch.stack(rew,dim=0),torch.stack(done,dim=0),new_info
    def close(self):
        del self.handle
@ray.remote(num_cpus=3)
class envir1(gym.core.Env):
    # def init_dri(self,height,width,driver_num,sampled_min_minutes,sampled_max_minutes,default_reward=-1):
    #     inits=self.history[['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude','minutes_in_day']].copy(deep=True)
    #     idexs=(inits['minutes_in_day']>=sampled_min_minutes)&(inits['minutes_in_day']<=sampled_max_minutes)
    #     inits=inits[idexs][:].reset_index(drop=True)
    #     inits=inits[['pickup_latitude','pickup_longitude']].values
    #     assert inits.shape[0]>=driver_num
    #     idx=np.random.choice(a=range(inits.shape[0]),size=driver_num,replace=False)
    #     matrix=np.zeros((height,width),dtype=int)
    #     for id in idx:
    #         start_block_la=int(np.round((inits[id,0]-configure.min_la)/configure.step_dis))
    #         start_block_lo=int(np.round((inits[id,1]-configure.min_lo)/configure.step_dis))
    #         matrix[start_block_la,start_block_lo]+=1
    #     return torch.from_numpy(matrix).float()
    def get_order(self,q,quos,global_dri,init_block,ending_block,running_dict,expEndTime,cost,height,width,dist_max,self_agent=[0,1],self_cost_rate=configure.cost_rate):
        alo=self.alo
        selected=[]
        getOrder=-1
        q_init=q
        quos_init=quos
        quos=quos.clone().numpy()
        q=q.clone().numpy()
        #q=quos[np.random.choice(a=range(len(quos)),size=1,replace=False)[0]]+np.random.normal(0,0.1)
        #q_init=torch.tensor([q],dtype=torch.float32)
        #print(q,quos)
        #q=np.array([q])
        quos=np.random.permutation(quos.T).T
        quotations=np.concatenate([q,quos],axis=0)
        if alo!=0:
            selected=np.argpartition(quotations, alo)[:alo]
        else:
            prob=[]
            ps=np.argsort(quotations)
            for i in range(len(ps)):
                if ps[i]<7:
                    prob.append((i,(7-ps[i])/7))
            for i in range(len(prob)):
                ran=np.random.rand()
                if ran<prob[i][1]:
                    selected.append(prob[i][0])
            selected=np.array(selected)
        #print(selected)
        def random_select(selected,selected_num):
            return np.random.choice(selected,p=selected_num/selected_num.sum(),size=1)[0]
        dri_block=init_block
        dist=0
        find=False
        prob_vec=np.zeros_like(selected,dtype=float)
        searched=[]
        while not find:
            searched=[]
            for f in range(dist):
                o=dist-f
                for _t in iter([(o,f),(o,-f),(-o,f),(-o,-f)]):
                    (i,j)=_t
                    dri_block=(init_block[0]+i,init_block[1]+j)
                    if dri_block[0]<0 or dri_block[0]>=height or dri_block[1]<0 or dri_block[1]>=width:
                        continue
                    if dri_block not in searched:
                        searched.append(dri_block)
                        prob_vec+=np.array([global_dri[s][dri_block[0],dri_block[1]] for s in selected])
            if np.sum(prob_vec)>0:
                find=True
            if find:
                getOrder=random_select(selected=selected,selected_num=prob_vec)
                break
            dist+=1
            if dist>dist_max:
                return torch.tensor([0.],dtype=torch.float32),False
        for s in searched:
            if global_dri[getOrder][s[0],s[1]]>0:
                global_dri[getOrder][s[0],s[1]]-=1
                break
        if expEndTime in running_dict.keys():
            running_dict[expEndTime].append((ending_block,getOrder))
        else:
            running_dict[expEndTime]=[(ending_block,getOrder)]
        if getOrder in self_agent:
            return q_init-cost*self_cost_rate[getOrder],getOrder
        else:
            return torch.tensor([0.],dtype=torch.float32),None
        
    def act_to_quo(self,quos,act):
        less=act<0
        bigger=act>=0
        q=less*((quos[1]-quos[0])*act+quos[1])
        q+=bigger*((quos[-1]-quos[0])*act+quos[0])
        return q
        # quos:[b,5]
        # segmental linear interpolation [-1,-0.5,0,0.5,1]
        # q=0
        
        # if -1 <= act and act < -0.5:
        #     q=quos[0]+(quos[1]-quos[0])*2*(act+1)
        # elif -0.5 <= act and act < 0:
        #     q=quos[1]+(quos[2]-quos[1])*2*(act+0.5)
        # elif 0 <= act and act < 0.5:
        #     q=quos[2]+(quos[3]-quos[2])*2*(act)
        # elif 0.5 <= act and act <= 1:
        #     q=quos[3]+(quos[4]-quos[3])*2*(act-0.5)
        # return torch.tensor([q],dtype=torch.float32)
    # def __init__(self,driver_nums,cost_rate,device,envid,data_key,start_index,end_index,alo):
    def init_dri(self,dri_key):
        self.global_dri=dri_key
    def to_block(self,la,lo):
        return int(np.round((la-configure.min_la)/configure.step_dis)),int(np.round((lo-configure.min_lo)/configure.step_dis))
    def _all_dri(self):
        return self.alldri
    def __init__(self,alo,per,start,end,data_keys,cost_rate,device,dist_max,agents_nums,numagents=10): 
        self.reqdata,self.qQuanhandle,self.timerec,self.zdata=ray.get(data_keys)
        self.reqdata,self.qQuanhandle,self.timerec,self.zdata=self.reqdata.copy(deep=True),self.qQuanhandle.clone(),self.timerec.clone(),self.zdata.clone()
        self.start=start
        self.end=end
        self.dist_max=dist_max
        reqdata_start_index=self.reqdata['abs_minutes']==(self.reqdata['abs_minutes'].iloc[0]+start)
        reqdata_end_index=self.reqdata['abs_minutes']==(self.reqdata['abs_minutes'].iloc[0]+end)
        reqdata_start=self.reqdata[reqdata_start_index].index[0]
        reqdata_end=self.reqdata[reqdata_end_index].index[0]-1
        self.start_index=reqdata_start
        self.end_index=reqdata_end
        self.alldri=(self.end_index-self.start_index)//2
        self.reqdata=self.reqdata[self.start_index:self.end_index]
        self.agent_nums=agents_nums
        self.device=device
        self.cost_rate=cost_rate
        self.cum_reward=0
        self.hour_reward=0
        self.stepnums=0
        
        
        self.index=0
        self.getted_req=0
        self.alo=alo
        self.minutes_start=0
        self.begin=self.reqdata['minutes_in_day'].iloc[0]
        
    def reqhandle(self):
        req=self.reqdata.iloc[self.stepnums]
        zdata=None
        qQuanhandle=self.qQuanhandle[self.minutes_now-self.minutes_start]
        self.stepnums+=1
        return [req,zdata,qQuanhandle]
    def transfer(self,obs):
        req,zdata,qQuanhandle=obs
        return torch.tensor(req['quantities']),qQuanhandle,torch.from_numpy(req[['pmin','pmid','pmax']].values),torch.tensor(req['fare_amonut'],dtype=torch.float32)
        #return torch.tensor(req['quantities']),qQuanhandle,torch.from_numpy(req[['q0','q0.25','q0.5','q0.75','q1']].values),torch.tensor(req['fare_amonut'],dtype=torch.float32)
    def reset(self):
        
        self.running_dri_dict={}
        self.getted_req=0
        self.minutes_now=self.start
        self.stepnums=0
        self.cum_reward=0
        self.last_obs=None
        infoma=self.reqhandle()
        self.last_obs=infoma
        self.minutes_start=self.reqdata['abs_minutes'].iloc[0]
        return self.transfer(infoma)
        '''
        try:
            self.minutes=0
            self.index=0
            self.getted_req=0
            data=self.data_key.copy(deep=True)
            begin=self.envid*60
            data=pd.DataFrame(data.iloc[self.start_index:self.end_index+1,:],copy=True)
            data.reset_index(drop=True,inplace=True)
            self.reqdataloader=datahandler.requesthandler(data)
            self.reqdataloader=torch.utils.data.DataLoader(self.reqdataloader,batch_size=1,shuffle=False,num_workers=0)
            self._dhand=self.iterhandler()
            self.running_dri_dict={}
            global_dri=[self.init_dri(configure.input_size[0],configure.input_size[1],dris,begin-30,begin) for dris in self.driver_nums]
            self.global_dri=torch.stack(global_dri,dim=0)
            self.last_minuts=self.envid*60-1
            self.cum_reward==torch.tensor([[0.]],dtype=torch.float32)
            self.hour_reward==torch.tensor([[0.]],dtype=torch.float32)
            self.stepnums=0
            next=self.getnext()
            self.last_obs=next
            return next
        except:
            return self.reset()
    '''
    def get_driver_nums(self):
        return [torch.sum(self.global_dri[i]).item() for i in range(self.global_dri.shape[0])]
    def step(self,actx):
        act,useless_=actx
        for key in list(self.running_dri_dict.keys()):
            if key<=self.minutes_now:
                for dria in self.running_dri_dict[key]:
                    ending_block,getOrder=dria
                    try:
                        self.global_dri[getOrder][ending_block[0],ending_block[1]]+=1
                    except:
                        print(self.global_dri)
                        print(ending_block)
                        print(getOrder)
                        raise Exception
                del self.running_dri_dict[key]
        #quos=self.last_obs[0][['quotation'+str(i) for i in range(10)]].values
        quos=self.last_obs[0][['quotation'+str(i) for i in range(10)]].values
        preds=self.last_obs[0][['pmin','pmid','pmax']].values
        #preds=self.last_obs[0][['q0','q0.25','q0.5','q0.75','q1']].values
        q=[self.act_to_quo(preds,act[i]) for i in range(act.shape[0])]
        #print(quos,preds,q,act)
        q=torch.concat(q,dim=0)
        quos=torch.tensor(quos,dtype=torch.float32)
        start_block=self.to_block(self.last_obs[0]['pickup_latitude'],self.last_obs[0]['pickup_longitude'])
        end_block=self.to_block(self.last_obs[0]['dropoff_latitude'],self.last_obs[0]['dropoff_longitude'])
        exendtime=int(self.last_obs[0]['abs_end_minutes'])
        lamb,get=self.get_order(
            q,quos,self.global_dri,start_block,end_block,self.running_dri_dict,exendtime,cost=self.transfer(self.last_obs)[3],height=configure.input_size[0],width=configure.input_size[1],dist_max=self.dist_max,self_agent=[i for i in range(self.agent_nums.shape[0])]
        )
        rewards=lamb[0]
        done=False
        if get != None:
            self.getted_req+=1
        self.minutes_now=int(self.last_obs[0]['abs_minutes'])
        self.cum_reward+=rewards
        self.hour_reward+=rewards
        obs_next=self.reqhandle()
        if self.stepnums==len(self.reqdata):
            done=True
        self.last_obs=obs_next
        if obs_next==None:
            done=True
        return [self.transfer(obs_next),(rewards,get),torch.tensor([done],dtype=bool),{'cum_reward':self.cum_reward,'hour_reward':self.hour_reward,'done':self.stepnums,'all':len(self.reqdata),'driver_nums':self.get_driver_nums(),'minutes':self.minutes_now,'running_dri':len(self.running_dri_dict),'getted_req':self.getted_req}]
    '''
    def step(self,act_obs):
        act,obs=act_obs
        obs=self.last_obs
        self.stepnums+=1
        done=False
        mats,feas,quos,preds,cost,dri=obs
        for key in list(self.running_dri_dict.keys()):
            if key<=self.last_minuts:
                for dria in self.running_dri_dict[key]:
                    ending_block,getOrder=dria
                    self.global_dri[getOrder][ending_block[0],ending_block[1]]+=1
                del self.running_dri_dict[key]
        q=self.act_to_quo(quos=preds,act=act)
        lamb,get=self.get_order(q[0],quos[0],self.global_dri,torch.nonzero(mats[0,0,:,:])[0].tolist(),torch.nonzero(mats[0,1,:,:]).tolist(),
            running_dict=self.running_dri_dict,expEndTime=int(feas[0,2].item()//60+self.last_minuts+0.99),cost=cost,height=configure.input_size[0],
            width=configure.input_size[1])
        rewards=lamb[0]
        if get:
            self.getted_req+=1
        if self.stepnums==len(self.reqdataloader):
            done=True
        self.last_minuts=int(feas[0,1].item())
        obs_next=self.getnext()
        self.cum_reward+=rewards
        self.hour_reward+=rewards
        self.last_obs=obs_next
        if obs_next==None:
            done=True
        return [obs_next,rewards,torch.tensor([done],dtype=bool),{'cum_reward':self.cum_reward,'hour_reward':self.hour_reward,'done':self.stepnums,'all':len(self.reqdataloader),'driver_nums':self.get_driver_nums(),'minutes':self.last_minuts,'running_dri':len(self.running_dri_dict),'getted_req':self.getted_req}]
    '''
class CarPoolEnv1(gym.Env):
    def __init__(self,args_list,device,history_data,agent_nums=np.array([1],dtype=np.float32),dist_max=5):
        self.device=device
        self.handle=[]
        self.args_list=args_list
        temp=[]
        self.init_dri=[]
        self.history_data=history_data
        self.dist_max=dist_max
        self.agent_nums=agent_nums
        assert self.agent_nums.sum()==1
        for args in args_list:
            self.handle.append(envir1.remote(*args,dist_max,agent_nums))
            temp.append(self.handle[-1]._all_dri.remote())
        temp=ray.get(temp)
        _=len(temp)
        for i in range(_):
            per=args_list[i][1]
            random=np.random.rand(10)
            random=random/(np.sum(random)+per)
            all=temp[i]
            dri_nums=[int(per*all*percents) for percents in self.agent_nums.tolist()]+[int(random[i]*all) for i in range(10)]
            self.init_dri.append(
                torch.stack([self.making_drivers(args_list[i][2]-30,args_list[i][2],dri_nums[j]) for j in range(len(dri_nums))])
            )

        self.dris=[]
        for i in range(len(self.init_dri)):
            self.dris.append(ray.put(self.init_dri[i]))
        self.done_list=[False for _ in range(len(self.handle))]
    def retran(self,obs):
        a=[[] for _ in range(len(obs[0]))]
        for i in range(len(obs)):
            for j in range(len(obs[i])):
                a[j].append(obs[i][j])
        return a
    
    def making_drivers(self,sampled_min_minutes,sampled_max_minutes,driver_num,height=configure.input_size[0],width=configure.input_size[1]):
        history=self.history_data        
    # def init_dri(self,height,width,driver_num,sampled_min_minutes,sampled_max_minutes,default_reward=-1):
        inits=history[['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude','minutes_in_day']].copy(deep=True)
        idexs=(inits['minutes_in_day']>=sampled_min_minutes)&(inits['minutes_in_day']<=sampled_max_minutes)
        inits=inits[idexs][:].reset_index(drop=True)
        inits=inits[['pickup_latitude','pickup_longitude']].values
        assert inits.shape[0]>=driver_num
        idx=np.random.choice(a=range(inits.shape[0]),size=driver_num,replace=False)
        matrix=np.zeros((height,width),dtype=int)
        for id in idx:
            start_block_la=int(np.round((inits[id,0]-configure.min_la)/configure.step_dis))
            start_block_lo=int(np.round((inits[id,1]-configure.min_lo)/configure.step_dis))
            matrix[start_block_la,start_block_lo]+=1
        return torch.from_numpy(matrix).float()
    def reset(self):
        obss=[]
        for i,env in enumerate(self.handle):
            key=self.dris[i]
            env.init_dri.remote(key)
            obss.append(env.reset.remote())
        obs2=[]
        obs2=ray.get(obss)
        obs2=self.retran(obs2)
        return obs2
    def step(self,_):
        act,obs=_
        handle=[]
        res=[]
        for i in range(len(self.handle)):
            handle.append(self.handle[i].step.remote([act[i].unsqueeze(0),obs]))
        res=ray.get(handle)
        new_obs=[]
        rew=[]
        done=[]
        info=[]
        for i in range(len(self.handle)):
            new_obs.append(res[i][0])
            rew.append(res[i][1])
            done.append(res[i][2])
            info.append(res[i][3])
        new_obs=self.retran(new_obs)
        new_info={
            'cum_reward':np.array([info[i]['cum_reward'].item() for i in range(len(self.handle))]),
            'hour_reward':np.array([info[i]['hour_reward'].item() for i in range(len(self.handle))]),
            'done':np.array([info[i]['done'] for i in range(len(self.handle))]),
            'all':np.array([info[i]['all'] for i in range(len(self.handle))]),
            'driver_nums':np.array([info[i]['driver_nums'] for i in range(len(self.handle))]),
            'minutes':np.array([info[i]['minutes'] for i in range(len(self.handle))]),
            'running_dri':np.array([info[i]['running_dri'] for i in range(len(self.handle))]),
            'getted_req':np.array([info[i]['getted_req'] for i in range(len(self.handle))])
        }
        return new_obs,rew,torch.stack(done,dim=0),new_info
    def close(self):
        del self.handle

register(
    id='CarPool-v0',
    entry_point='utils:CarPoolEnv'
)
register(
    id='CarPool-v1',
    entry_point='utils:CarPoolEnv1'
)
