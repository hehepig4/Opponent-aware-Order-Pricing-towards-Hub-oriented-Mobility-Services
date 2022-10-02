import configure
import predic_model as pmodel
import dataloader as dloader
import torch
from torch.utils.data import DataLoader
import torch.utils.tensorboard as tb
import tqdm
import sys
import numpy as np
import pandas as pd
from torch.nn import MSELoss
@torch.no_grad()

    
def eval(modelset,res,first,dset):
    dataset=dset

    
    '''model=pmodel.predic_model_2(
        kernel_size=(7,7),
        input_dim=2,
        hidden_dim=4,
        linear_dim=[1024,256,128],
        convlstm_linear_size=1024,
        var_num=5,
        output_dim=3,
        input_height=configure.input_size[0],
        input_width=configure.input_size[1],
        seq_len=configure.seq_len
    ).cuda()
    '''
    

    models={}
    picp={}
    pinaw={}
    cwc={}
    ais={}
    real_minrate={}
    real_maxrate={}
    model_names=[]
    min_mse={}
    max_mse={}
    mean_mse={}
    alpha=0.1
    mul=1-alpha

    result=pd.DataFrame(columns=['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude','minutes_in_day','weekday','trip_time_in_secs','trip_distance']+['quotation'+str(i) for i in range(10)])

    for mm in modelset:
        save_path=mm[0]
        model_name=mm[1]
        modelis=mm[2]
        model=modelis
        model=torch.nn.DataParallel(model,device_ids=[0,1])
        model.load_state_dict(torch.load(save_path)['model_state_dict'])
        model.eval()
        
        models.update({model_name:model})
        picp.update({model_name:0})
        pinaw.update({model_name:0})
        cwc.update({model_name:0})
        real_maxrate.update({model_name:0})
        real_minrate.update({model_name:0})
        min_mse.update({model_name:0})
        max_mse.update({model_name:0})
        mean_mse.update({model_name:0})
        model_names.append(model_name)
        result.insert(result.shape[1],model_name+'_min',0)
        result.insert(result.shape[1],model_name+'_mean',0)
        result.insert(result.shape[1],model_name+'_max',0)
        result.astype(float)
    dataloader=DataLoader(dataset,batch_size=16,shuffle=False,num_workers=8,pin_memory=True)
    num=0
    
    with tqdm.tqdm(total=len(dataloader)) as pbar:
        for (x1,x2,y1,y2,ori) in dataloader:
            pbar.update(1)
            x1,x2,y1,y2,ori=x1.cuda(),x2.cuda(),y1.cuda(),y2.cuda(),ori.cuda()
            with torch.no_grad():
                outputs={model_name:models[model_name](x1,x2) for model_name in model_names}
            for batch in range(x1.shape[0]):
                new_list=ori[batch,:].cpu().tolist()
                new_list.extend(y2[batch,:].cpu().tolist())
                for model_name in model_names:
                    new_list.append(outputs[model_name][batch,0].cpu().item())
                    new_list.append(outputs[model_name][batch,1].cpu().item())
                    new_list.append(outputs[model_name][batch,2].cpu().item())
                new_list=pd.DataFrame([new_list],columns=result.columns.values.tolist())
                result=pd.concat([result,new_list],ignore_index=True)
                
            for model_name in model_names:
                
                if_bigger=torch.lt(outputs[model_name][:,0],y2[:,0]).bool()
                if_less=torch.gt(outputs[model_name][:,-1],y2[:,-1]).bool()
                
                target_length=y2[:,-1]-y2[:,0]
                real_length=outputs[model_name][:,-1]-outputs[model_name][:,0]
                
                
                real_maxrate[model_name]+=if_bigger.int().sum().item()
                real_minrate[model_name]+=if_less.int().sum().item()
                
                for data in range(y2.shape[1]):
                    if_bigger=torch.lt(outputs[model_name][:,0],y2[:,data]).bool()
                    if_less=torch.gt(outputs[model_name][:,-1],y2[:,data]).bool()
                    ok=if_less & if_bigger
                    picp[model_name]+=ok.int().sum().item()
                    
                min_mse[model_name]+=torch.pow(outputs[model_name][ok,0]-y2[ok,0],2).sum().item()
                max_mse[model_name]+=torch.pow(outputs[model_name][ok,-1]-y2[ok,-1],2).sum().item()
                max_mse[model_name]+=torch.pow(outputs[model_name][ok,1]-y1[ok,1],2).sum().item()
                pinaw[model_name]+=torch.div(real_length,target_length).sum().item()

            num+=x1.shape[0]
        
            if num % 8000 == 1:
            #    result.to_hdf('result_'+pbar.n % 100000+'.h5','result')     
                pbar.write(model_name+'\n'+
              'num:'+str(num)+'\n'+
              'real_minrate:'+str(real_minrate[model_name]/num)+'\n'+
              'real_maxrate:'+str(real_maxrate[model_name]/num)+'\n'+
              'picp:'+str(picp[model_name]/(10*num))+'\n'+
              'pinaw:'+str(pinaw[model_name]/num)+'\n'+
              'cwc:'+str(cwc[model_name]/num)+'\n'+
              'min_mse:'+str(min_mse[model_name]/num)+'\n'+
              'max_mse:'+str(max_mse[model_name]/num)+'\n'+
              'mean_mse:'+str(mean_mse[model_name]/num)+'\n')
                
    for model_name in model_names:
        picp[model_name]=picp[model_name]/(10*num)
        pinaw[model_name]=pinaw[model_name]/num
        real_maxrate[model_name]=real_maxrate[model_name]/num
        real_minrate[model_name]=real_minrate[model_name]/num
        min_mse[model_name]=min_mse[model_name]/num
        max_mse[model_name]=max_mse[model_name]/num
        mean_mse[model_name]=mean_mse[model_name]/num
        gamma=0
        if picp[model_name]<mul:
            gamma=1
        eta=1
        cwc[model_name]=pinaw[model_name]*(1+gamma*np.exp(-eta*(picp[model_name]-mul)))
    for model_name in model_names:
        print(model_name+'\n'+
              'num:'+str(num)+'\n'+
              'real_minrate:'+str(real_minrate[model_name])+'\n'+
              'real_maxrate:'+str(real_maxrate[model_name])+'\n'+
              'picp:'+str(picp[model_name])+'\n'+
              'pinaw:'+str(pinaw[model_name])+'\n'+
              'cwc:'+str(cwc[model_name])+'\n'+
              'min_mse:'+str(min_mse[model_name])+'\n'+
              'max_mse:'+str(max_mse[model_name])+'\n'+
              'mean_mse:'+str(mean_mse[model_name])+'\n')
        print(model_name+
              'num:'+str(num)+'\n'+
              'real_minrate:'+str(real_minrate[model_name])+'\n'+
              'real_maxrate:'+str(real_maxrate[model_name])+'\n'+
              'picp:'+str(picp[model_name])+'\n'+
              'pinaw:'+str(pinaw[model_name])+'\n'+
              'cwc:'+str(cwc[model_name])+'\n'+
              'min_mse:'+str(min_mse[model_name])+'\n'+
              'max_mse:'+str(max_mse[model_name])+'\n'+
              'mean_mse:'+str(mean_mse[model_name])+'\n',file=open('result.txt','a'))
    return result,False
'''
eval_set=[
    ['checkpoint/p7.pth','MSE_loss',pmodel.predic_model_2(
        kernel_size=(7,7),
        input_dim=2,
        hidden_dim=4,
        linear_dim=[1024,256,256],
        convlstm_linear_size=1024,
        var_num=5,
        output_dim=3,
        input_height=configure.input_size[0],
        input_width=configure.input_size[1],
        seq_len=configure.seq_len
    ).cuda(),'p7'],
    ['checkpoint/p8.pth','MSE_loss_Rank_loss',pmodel.predic_model_2(
        kernel_size=(7,7),
        input_dim=2,
        hidden_dim=4,
        linear_dim=[1024,256,256],
        convlstm_linear_size=1024,
        var_num=5,
        output_dim=3,
        input_height=configure.input_size[0],
        input_width=configure.input_size[1],
        seq_len=configure.seq_len
    ).cuda(),'p8'],
    ['checkpoint/p9.pth','pinball_loss_Rank_loss',pmodel.predic_model_2(
        kernel_size=(7,7),
        input_dim=2,
        hidden_dim=4,
        linear_dim=[1024,256,256],
        convlstm_linear_size=1024,
        var_num=5,
        output_dim=3,
        input_height=configure.input_size[0],
        input_width=configure.input_size[1],
        seq_len=configure.seq_len
    ).cuda(),'p9'],['checkpoint/p10.pth','weighted_pinball_loss_Rank_loss',pmodel.predic_model_2(
        kernel_size=(7,7),
        input_dim=2,
        hidden_dim=4,
        linear_dim=[1024,256,256],
        convlstm_linear_size=1024,
        var_num=5,
        output_dim=3,
        input_height=configure.input_size[0],
        input_width=configure.input_size[1],
        seq_len=configure.seq_len
    ).cuda(),'p10']
]
'''
def real_fare(subset):
    dataloader=DataLoader(subset,batch_size=16,shuffle=False,num_workers=8,pin_memory=True)
    result=pd.DataFrame(columns=['fare_amount'])
    with tqdm.tqdm(total=len(dataloader)) as pbar:
        for fare_a in dataloader:
            for batch in range(fare_a.shape[0]):
                result.loc[result.index.size]=fare_a[batch].item()
            pbar.update(1)
    return result
if __name__=='__main__':
    dloader._init()
    data=[]
    data.extend([50000*i for i in range(110)])
    dataset=dloader.streamingdataset5(
        start=0,
        end=data[-1]+configure.seq_len
    )
    subsets=[
        torch.utils.data.Subset(dataset,list(range(data[i],data[i+1]))) for i in range(len(data)-1)
    ]
    result=None
    first=True
    for i in range(0,len(data)-1):
        result=real_fare(subsets[i])
        result.to_hdf('fare'+str(i)+'.h5','fare')
        #result,first=eval(eval_set,result,first,subsets[i])
        #result.to_hdf('result'+str(i)+'.h5','result',mode='w')