import numpy as np
import ray
import torch
import configure
@ray.remote(num_cpus=1)
class greedAgent():
    def __init__(self):
        self.reward_dict={}
        self.action_list=[]
        self.get=0
        pass
    def act(self,Q,Q_dist,quos,cost,info=None):
        return torch.tensor([-1.])
    def add_reward(self,minutes,reward,act,ifget):
        if minutes not in self.reward_dict.keys():
            self.reward_dict[minutes]=reward
        else:
            self.reward_dict[minutes]+=reward
        self.action_list.append(act)
        if ifget:
            self.get+=1
    def updating(self,trans):
        pass
@ray.remote(num_cpus=1)
class randomAgent():
    def __init__(self):
        self.reward_dict={}
        self.action_list=[]
        self.get=0
        pass
    def act(self,Q,Q_dist,quos,cost,info=None):
        rand=np.random.rand()
        rand=(rand-0.5)*2
        rand=torch.tensor([rand],dtype=torch.float32)
        return rand
    def add_reward(self,minutes,reward,act,ifget):
        if minutes not in self.reward_dict.keys():
            self.reward_dict[minutes]=reward
        else:
            self.reward_dict[minutes]+=reward
        self.action_list.append(act)
        if ifget:
            self.get+=1
    def updating(self,trans):
        pass
@ray.remote(num_cpus=1)
class mutgreedAgent():
    def __init__(self):
        self.reward_dict={}
        self.action_list=[]
        self.get=0
        pass
    def act(self,Q,Q_dist,quos,cost,info=None):
        return torch.tensor([-1.])
    #   return torch.tensor([-1.,-1.])
    def add_reward(self,minutes,reward,act,ifget):
        if minutes not in self.reward_dict.keys():
            self.reward_dict[minutes]=reward
        else:
            self.reward_dict[minutes]+=reward
        self.action_list.append(act)
        if ifget:
            self.get+=1
    def updating(self,trans):
        pass
@ray.remote(num_cpus=1)
class multiArmBanditAgent():
    def __init__(self,arms):
        self.reward_dict={}
        self.arms=arms
        self.arms_success=[0 for i in range(arms)]
        self.arms_success=np.array(self.arms_success)
        self.arms_fail=[0 for i in range(arms)]
        self.arms_fail=np.array(self.arms_fail)
        self.arms_quantile=[0]+[1/(arms-1) for i in range(1,arms)]
        self.arms_quantile=np.array(self.arms_quantile)
        self.arms_quantile=np.cumsum(self.arms_quantile)-0.5
        self.arms_quantile=self.arms_quantile*2
        self.arms_quantile=torch.tensor(self.arms_quantile,dtype=torch.float32)
       
    def get_probs(self):
        probs=[]
        for i in range(self.arms):
            prob=np.random.beta(self.arms_success[i]+1,self.arms_fail[i]+1)
            probs.append(prob)
        return np.array(probs)
    def act(self,Q,Q_dist,quos,cost,info=None):
        probs=self.get_probs()
        probs=torch.from_numpy(probs)
        temp=[]
        for q in range(len(self.arms_quantile)):
            temp.append(self.act_to_quo(quos,self.arms_quantile[q]))
        act_quos=torch.tensor(temp,dtype=torch.float32)
        vls=act_quos
        vls=probs*vls
        act=np.argmax(vls)
        return torch.tensor([self.arms_quantile[act].item()])
    def log(self):
        pass
    def act_to_quo(self,quos,act):
        less=act<0
        bigger=act>=0
        q=less*((quos[1]-quos[0])*act+quos[1])
        q+=bigger*((quos[-1]-quos[0])*act+quos[0])
        return q
    def updating(self,transition):
        obs,acts,obs_next,rewards,done=transition
        arms=torch.argwhere(acts==self.arms_quantile)
        if rewards[1]==0:
            self.arms_success[arms]+=1
        else:
            self.arms_fail[arms]+=1
@ray.remote(num_cpus=1)
class multiTomsArmBanditAgent():
    def __init__(self,arms,num_agents=2):
        self.reward_dict={}
        self.arms=arms
        self.arms_success=[0 for i in range(arms)]
        self.arms_success=[np.array(self.arms_success) for i in range(num_agents)]
        self.arms_fail=[0 for i in range(arms)]
        self.arms_fail=[np.array(self.arms_fail) for i in range(num_agents)]
        self.arms_quantile=[0]+[1/(arms-1) for i in range(1,arms)]
        self.arms_quantile=np.array(self.arms_quantile)
        self.arms_quantile=np.cumsum(self.arms_quantile)-0.5
        self.arms_quantile=self.arms_quantile*2
        self.arms_quantile=torch.tensor(self.arms_quantile,dtype=torch.float32)
        self.cost_rate=configure.cost_rate
        self.num_agents=num_agents
    def get_probs(self,index):
        probs=[]
        for i in range(self.arms):
            prob=np.random.beta(self.arms_success[index][i]+1,self.arms_fail[index][i]+1)
            probs.append(prob)
        return np.array(probs)
    def act(self,Q,Q_dist,quos,cost,info=None):
        ans=[]
        for i in range(self.num_agents):
            probs=self.get_probs(i)
            probs=torch.from_numpy(probs)
            temp=[]
            for q in range(len(self.arms_quantile)):
                temp.append(self.act_to_quo(quos,self.arms_quantile[q]))
            act_quos=torch.tensor(temp,dtype=torch.float32)
            vls=act_quos-cost*self.cost_rate[i]
            vls=probs*vls
            act=np.argmax(vls)
            ans.append(self.arms_quantile[act])
        return torch.tensor(ans)
    def log(self):
        pass
    def act_to_quo(self,quos,act):
        less=act<0
        bigger=act>=0
        q=less*((quos[1]-quos[0])*act+quos[1])
        q+=bigger*((quos[-1]-quos[0])*act+quos[0])
        return q
    def updating(self,transition):
        obs,acts,obs_next,rewards,done=transition
        for i in range(acts.shape[0]):
            arms=torch.argwhere(acts[i]==self.arms_quantile)
            if rewards[1]==i:
                self.arms_success[i][arms]+=1
            else:
                self.arms_fail[i][arms]+=1
@ray.remote(num_cpus=1)
class multiQuoMultiArmBanditAgent():
    def __init__(self,arms,agents_num=len(configure.cost_rate)):
        self.arms=arms
        self.agents_num=agents_num
        self.counts=[torch.zeros(arms,dtype=torch.float32) for _ in range(agents_num)]
        self.values=[torch.zeros(arms,dtype=torch.float32) for _ in range(agents_num)]
        self.arms_quantile=[0]+[1/(arms-1) for i in range(1,arms)]
        self.arms_quantile=np.array(self.arms_quantile)
        self.arms_quantile=np.cumsum(self.arms_quantile)-0.5
        self.arms_quantile=self.arms_quantile*2
        self.arms_quantile=torch.tensor(self.arms_quantile,dtype=torch.float32)
        self.cost_rate=configure.cost_rate
    def act_to_quo(self,quos,act):
        less=act<0
        bigger=act>=0
        q=less*((quos[1]-quos[0])*act+quos[1])
        q+=bigger*((quos[-1]-quos[0])*act+quos[0])
        return q
    def act(self,Q,Q_dist,quos,cost,info=None):
        temp=[]
        for q in range(len(self.arms_quantile)):
            temp.append(self.act_to_quo(quos,self.arms_quantile[q]))
        act_quos=torch.tensor(temp,dtype=torch.float32)
        act=[]
        for i in range(self.agents_num):
            if torch.any(self.counts[i])==0:
                act.append(self.arms_quantile[torch.argwhere(self.counts[i]==0)[0]])
            else:
                ucbs=self.get_ucb(i)
                values=act_quos-cost*self.cost_rate[i]
                a=torch.argmax(ucbs*values)
                act.append(self.arms_quantile[a])
        return torch.tensor(act)
    def get_ucb(self,i):
        total=self.counts[i].sum()
        constant=torch.broadcast_to(total,self.counts[i].shape)
        bonus=torch.sqrt(torch.log(constant)/self.counts[i])
        return self.values[i]+bonus
    def updating(self,transition):
        obs,acts,obs_next,rewards,done=transition
        
        for i in range(acts.shape[0]):

            arms=torch.argwhere(acts[i]==self.arms_quantile)
            self.counts[i][arms]=self.counts[i][torch.argwhere(acts[i]==self.arms_quantile)]+1
            if rewards[1]==i:
                self.values[i][arms]=(self.counts[i][arms]/(1+self.counts[i][arms]))*self.values[i][arms]+(1/(1+self.counts[i][arms]))*1
            else:
                self.values[i][arms]=(self.counts[i][arms]/(1+self.counts[i][arms]))*self.values[i][arms]+(1/(1+self.counts[i][arms]))*0
@ray.remote(num_cpus=1)
class estimatemutiArmBanditAgent():
    def __init__(self,arms):
        self.reward_dict={}
        self.arms=arms
        self.arms_success=[0 for i in range(arms)]
        self.arms_success=np.array(self.arms_success)
        self.arms_fail=[0 for i in range(arms)]
        self.arms_fail=np.array(self.arms_fail)
        self.arms_quantile=[0]+[1/(arms-1) for i in range(1,arms)]
        self.arms_quantile=np.array(self.arms_quantile)
        self.arms_quantile=np.cumsum(self.arms_quantile)-0.5
        self.arms_quantile=self.arms_quantile*2
        self.arms_quantile=torch.tensor(self.arms_quantile,dtype=torch.float32)
    def get_probs(self):
        probs=[]
        for i in range(self.arms):
            prob=np.random.beta(self.arms_success[i]+1,self.arms_fail[i]+1)
            probs.append(prob)
        return np.array(probs)
    def act(self,Q,Q_dist,quos,cost,info=None):
        if self.policy(Q_dist,Q):
            probs=self.get_probs()
            probs=torch.from_numpy(probs)
            temp=[]
            for q in range(len(self.arms_quantile)):
                temp.append(self.act_to_quo(quos,self.arms_quantile[q]))
            act_quos=torch.tensor(temp,dtype=torch.float32)
            vls=act_quos
            vls=probs*vls
            act=np.argmax(vls)
            act=self.arms_quantile[act]
            return act
        else:
            return self.arms_quantile[np.random.randint(0,self.arms)]
    def log(self,arg_str):
        return self.get_probs()
    def act_to_quo(self,quos,act):
        less=act<0
        bigger=act>=0
        q=less*((quos[1]-quos[0])*act+quos[1])
        q+=bigger*((quos[-1]-quos[0])*act+quos[0])
        return q
    def updating(self,transition):
        obs,acts,obs_next,rewards,done=transition
        arms=torch.argwhere(acts==self.arms_quantile)
        if rewards>0:
            self.arms_success[arms]+=1
        else:
            self.arms_fail[arms]+=1
        
    def est_toler(self):
        probs=self.get_probs()
        toler=probs/np.max(probs)
        return toler,np.max(probs)
    def policy(self,Q_dist,Q):
        Q_dist=Q_dist[0]
        toler,estm=self.est_toler()
        #print(Q_dist)
        #print(toler)
        #print(estm)
        selected_rate=np.sum(toler)/self.arms
        allow=min(selected_rate,1)
        allow=1-allow
        #print(allow)
        #print(selected_rate)
        #print(Q)
        rank=0
        while not rank==len(Q_dist)-1 and not (Q>Q_dist[rank] and Q<=Q_dist[rank+1]) :
            rank+=1
        rankquan=0
        if rank==len(Q_dist)-1:
            rankquan=1
        else:
            rankquan=configure.Qquan[rank]
        #print(rankquan)
        if rankquan>allow:
            return True
        return False
@ray.remote(num_cpus=1)
class LinTSMultiArmbanditAgent():
    def __init__(self,arms,dimension,agents_num=[0],delta=0.5,epsilon=1/np.log(20000),R=0.01):
        self.v=[R*np.sqrt(24/epsilon*dimension*np.log(1/delta)) for i in range(arms)]
        self.B=[np.identity(dimension) for i in range(arms)]
        self.mu_hat=[np.zeros((dimension,1)) for i in range(arms)]
        self.f=[np.zeros((dimension,1)) for i in range(arms)]
        self.arms=arms
        self.arms_quantile=[0]+[1/(arms-1) for i in range(1,arms)]
        self.arms_quantile=np.array(self.arms_quantile)
        self.arms_quantile=np.cumsum(self.arms_quantile)-0.5
        self.arms_quantile=self.arms_quantile*2
        self.arms_quantile=torch.tensor(self.arms_quantile,dtype=torch.float32)
        self.last_arm=None
        self.last_context=None
        self.agents_num=agents_num
    def sampled(self,context,index):
        v=self.v[index]
        Bm=self.B[index]
        mu_hat=self.mu_hat[index]
        f=self.f[index]
        param1=np.matmul(context.T,mu_hat)
        param2=v**2*np.matmul(np.matmul(context.T,np.linalg.inv(Bm)),context)
        return np.random.normal(param1,param2)
    def act(self,Q,Q_dist,quos,cost,info=None):
        assert info != None
        context=np.concatenate([quos.numpy(),cost.unsqueeze(0).numpy(),info],axis=0).reshape(-1,1)
        temp=np.zeros((self.arms))
        for i in range(self.arms):
            temp[i]=self.sampled(context,i)
        self.last_arm=np.argmax(temp)
        self.last_context=context
        return self.arms_quantile[self.last_arm].unsqueeze(0)
    def updating(self,transition):
        obs,acts,obs_next,rewards,done=transition
        rew=torch.tensor([0.])
        if rewards[1] in self.agents_num:
            rew=rewards[0]
        arm=self.last_arm
        context=self.last_context
        self.B[arm]+=np.matmul(context,context.T)
        self.f[arm]+=context*rew.numpy()
        self.mu_hat[arm]=np.matmul(np.linalg.inv(self.B[arm]),self.f[arm])
@ray.remote(num_cpus=1)
class MulLinTSMultiArmbanditAgent():
    class subLinTSMultiArmbanditAgent():
        def __init__(self,arms,dimension,agents_num=[0],delta=0.5,epsilon=1/np.log(20000),R=0.01):
            self.v=[R*np.sqrt(24/epsilon*dimension*np.log(1/delta)) for i in range(arms)]
            self.B=[np.identity(dimension) for i in range(arms)]
            self.mu_hat=[np.zeros((dimension,1)) for i in range(arms)]
            self.f=[np.zeros((dimension,1)) for i in range(arms)]
            self.arms=arms
            self.arms_quantile=[0]+[1/(arms-1) for i in range(1,arms)]
            self.arms_quantile=np.array(self.arms_quantile)
            self.arms_quantile=np.cumsum(self.arms_quantile)-0.5
            self.arms_quantile=self.arms_quantile*2
            self.arms_quantile=torch.tensor(self.arms_quantile,dtype=torch.float32)
            self.last_arm=None
            self.last_context=None
            self.agents_num=agents_num
        def sampled(self,context,index):
            v=self.v[index]
            Bm=self.B[index]
            mu_hat=self.mu_hat[index]
            f=self.f[index]
            param1=np.matmul(context.T,mu_hat)
            param2=v**2*np.matmul(np.matmul(context.T,np.linalg.inv(Bm)),context)
            return np.random.normal(param1,param2)
        def act(self,Q,Q_dist,quos,cost,info=None):
            assert info != None
            context=np.concatenate([quos.numpy(),cost.unsqueeze(0).numpy(),info],axis=0).reshape(-1,1)
            temp=np.zeros((self.arms))
            for i in range(self.arms):
                temp[i]=self.sampled(context,i)
            self.last_arm=np.argmax(temp)
            self.last_context=context
            return self.arms_quantile[self.last_arm].unsqueeze(0)
        def updating(self,transition):
            obs,acts,obs_next,rewards,done=transition
            rew=0
            if rewards[1] in self.agents_num:
                rew=rewards[0].numpy()
            arm=self.last_arm
            context=self.last_context
            self.B[arm]+=np.matmul(context,context.T)
            self.f[arm]+=context*rew
            self.mu_hat[arm]=np.matmul(np.linalg.inv(self.B[arm]),self.f[arm])
    def __init__(self,arms,dimension,agents_num=[0,1],delta=0.5,epsilon=1/np.log(20000),R=0.01):
        self.single=[self.subLinTSMultiArmbanditAgent(arms,dimension,[agents_num[i]],delta,epsilon,R) for i in range(2)]
        self.agents_num=agents_num
    def act(self,Q,Q_dist,quos,cost,info=None):
        acts=[]
        for i in range(len(self.agents_num)):
            acts.append(self.single[i].act(Q,Q_dist,quos,cost,info[i:i+1]))
        return torch.cat(acts,dim=0)
    def updating(self,transition):
        obs,acts,obs_next,rewards,done=transition
        for i in range(len(self.agents_num)):
            self.single[i].updating((obs,acts[i],obs_next,rewards,done))
class multiAgents():
    def __init__(self,agents,arg_str):
        self.agents=agents
        self.arg_str=arg_str
    def acts(self,Q,Q_dist,quos,cost,info=None):
        handle=[]
        for i,agent in enumerate(self.agents):
            dri_nums=info['driver_nums'][i]
            handle.append(agent.act.remote(Q[i],Q_dist[i],quos[i],cost[i],dri_nums[0:len(configure.cost_rate)]))
        actions=ray.get(handle)
        return torch.stack(actions)
    def updating(self,transitions):
        for i,agent in enumerate(self.agents):
            agent.updating.remote(transitions[i])
    def log(self,writter):
        for i,agent in enumerate(self.agents):
            agent.log.remote(self.arg_str[i],writter)
        