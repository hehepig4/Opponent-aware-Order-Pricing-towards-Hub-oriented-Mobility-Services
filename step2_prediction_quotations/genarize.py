'''import configure
import predic_model as pmodel

import torch
from torch.utils.data import DataLoader
import torch.utils.tensorboard as tb
import tqdm
import pandas as pd

import torch
import torch.utils.data as pytorch_data
import numpy as np
import configure
import pandas as pd
import queue


def _init():
    global _global_dict
    try:
        _global_dict
    except:
        _global_dict = {}
def exist_key(key):
    return key in _global_dict
def set_value(key,value):
    try:
        _global_dict[key]=value
    except:
        _init()
def get_value(key):
    try:
        return _global_dict[key]
    except:
        if key=='file':
            set_value(key,pd.read_hdf('data.h5','data'))
        return _global_dict[key]
def read(s=0,e=None):
    data=get_value('file')
    
    #print(data.shape)
    return data.iloc[s:e]

min_la=configure.min_la
max_la=configure.max_la
min_lo=configure.min_lo
max_lo=configure.max_lo
step_dis=step_dis=configure.step_dis
startall=0
endall=0

import datamutiprocessing as dmp
from multiprocessing import Process, Queue ,Manager
import threading
import time




def insert_thread(sindex=0,global_input_queue=None,global_output_queue=None,oridata=None):
    index_now=sindex
    while index_now<=endall-startall:
        data_index=oridata.index[index_now]
        data=return_oral_data(index_now,oridata)
        global_input_queue.put([data_index,data])
        index_now+=1
        
        
def start_backend(prepared_queue=None,global_input_queue=None,global_output_queue=None,oridata=None):
        processes=[]
        threads=[]
        threads.append(threading.Thread(target=insert_thread,args=(0,global_input_queue,global_output_queue,oridata,)))
        for i in range(4):
            processes.append(Process(target=dmp.backend_prepare_thread,args=(global_input_queue,global_output_queue,min_la,max_la,min_lo,max_lo,step_dis,)))
        for p in processes:
            p.start()
        for t in threads:
            t.start()
            
def return_oral_data(index,oridata=None):
        if index < configure.seq_len:
            return oridata.iloc[:index+1]
        else:
            return oridata.iloc[index-configure.seq_len+1:index+1]      

  
class iter_dataset(pytorch_data.IterableDataset):
    ''''''
    output: x=[batch,seq_len,]
            y=[tensor(quantiles)[quantiles_num]]
    ''''''
    def __init__(self,start=0,end=None,seq_len=configure.seq_len,output_queue=None):
        super(iter_dataset,self).__init__()
        self.oridata=read(start,end)
        global oridata,startall,endall
        oridata=self.oridata
        self.seq_len=seq_len
        self.length=(end-start)+1
        self.start=start
        startall=start
        self.end=end
        endall=end
        self.prepared_length=0
        self.max_data_length=1000
        time.sleep(1)
        self.nums=0
        self.output_queue=output_queue
    def __iter__(self):
        nums=0
        while True:
            nums+=1
            if nums>=endall-startall:
                break
            gtone=self.output_queue.get(True)
            yield gtone
        return StopIteration()
    
    def __len__(self):
        return self.length-1
odata=None
@torch.no_grad()
def eval(save_path=None,model_name=None,modelis=None,log_dir=None,prepared_queue=None,global_input_queue=None,global_output_queue=None,odata=None):
    odata.insert(odata.shape[1],model_name+'_model_predmin',0)
    odata.insert(odata.shape[1],model_name+'_model_predmid',0)
    odata.insert(odata.shape[1],model_name+'_model_predmax',0)
    dataset=iter_dataset(
        start=0,
        end=150000000,
        output_queue=global_output_queue)
    
    ''''''model=pmodel.predic_model_2(
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
    ''''''
    model=modelis
    model=torch.nn.DataParallel(model,device_ids=[0,1])
    model.load_state_dict(torch.load(save_path)['model_state_dict'])
    model.eval()
    dataloader=DataLoader(dataset,batch_size=32,shuffle=False)
    start_backend(prepared_queue=prepared_queue,global_input_queue=global_input_queue,global_output_queue=global_output_queue,oridata=odata)
    with tqdm.tqdm(total=150000000) as pbar:
        for (index,x1,x2,y1,y2) in dataloader:
            x1,x2,y1,y2=x1.cuda(),x2.cuda(),y1.cuda(),y2.cuda()
            output=model(x1,x2)
            for batch in range(output.shape[0]):
                odata.loc[index[batch].item(),'model_predmin']=output[batch,0].item()
                odata.loc[index[batch].item(),'model_predmid']=output[batch,1].item()
                odata.loc[index[batch].item(),'model_predmax']=output[batch,2].item()
                pbar.update(1)

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
    ).cuda(),'p9'],
    
]

if __name__=='__main__':
    handle=Manager()
    global_input_queue=handle.Queue(maxsize=10000)
    global_output_queue=handle.Queue(maxsize=100000)
    prepared_queue=queue.Queue(maxsize=100000)
    _init()
    odata=get_value('file')
    for i in range(len(eval_set)):
        eval(eval_set[i][0],eval_set[i][1],eval_set[i][2],eval_set[i][3],prepared_queue,global_input_queue,global_output_queue,odata)
    odata.to_hdf('data_final.h5','data')'''
import time
import configure
import predic_model as pmodel
import dataloader as dloader
import torch
from torch.utils.data import DataLoader
import torch.utils.tensorboard as tb
import tqdm
import sys
@torch.no_grad()
def eval(save_path=None,model_name=None,modelis=None,log_dir=None):
    dataset=dloader.streamingdataset4(
        start=0,
        end=150000)
   
    model=modelis
    model=torch.nn.DataParallel(model,device_ids=[0,1])
    model.load_state_dict(torch.load(save_path)['model_state_dict'])
    model.eval()
    
    dataloader=DataLoader(dataset,batch_size=32,shuffle=False,num_workers=32,pin_memory=True)
    
    with tqdm.tqdm(total=len(dataloader)) as pbar:
        for (x1,x2,y1,y2) in dataloader:
            with torch.no_grad():
                time.perf_counter()
                x1,x2=x1.cuda(),x2.cuda()
                modelis(x1,x2)
                print(time.perf_counter())
            '''
            x1,x2,y1,y2=x1.cuda(),x2.cuda(),y1.cuda(),y2.cuda()
            output=model(x1,x2)
            for batch in range(output.shape[0]):
                num+=1
                if output[batch,0].item()>output[batch,2].item():
                    invailrate+=1
                    invaliave+=y2[batch,-1].item()-y2[batch,0].item()
                    continue
                elif output[batch,0].item()>output[batch,1].item()or output[batch,2].item()<output[batch,1].item():
                    meancrossrate+=1
                if output[batch,0].item()<y2[batch].min().item():
                    minrate+=1
                if output[batch,2].item()>y2[batch].max().item():
                    maxrate+=1
                temp=0
                while temp<10 and output[batch,0].item()>y2[batch,temp].item():
                    temp+=1
                avemin+=temp+1
                temp=0
                while temp<10 and output[batch,2].item()>y2[batch,temp].item():
                    temp+=1
                avemax+=temp+1
                temp=1
                while temp<10 and output[batch,1].item()>y2[batch,temp].item():
                    temp+=1
                avemean+=temp+1
                cover_rate+=abs(output[batch,2]-output[batch,0])/abs(y2[batch,-1]-y2[batch,0])
            aveminmse+=torch.sum(torch.pow(output[:,0]-y1[:,0],2))
            avemaxmse+=torch.sum(torch.pow(output[:,2]-y1[:,2],2))
            aveminloss+=torch.sum(output[:,0]-y1[:,0])
            avemaxloss+=torch.sum(output[:,2]-y1[:,2])
            
            
            '''
            pbar.update(1)
            '''
            if pbar.n % 100==0:
                pbar.write('avemin:%.3f avemax:%.3f avemean:%.3f aveminloss:%.3f avemaxloss:%.3f '%(avemin/num,avemax/num,avemean/num,aveminloss/num,avemaxloss/num))
                pbar.write('minrate:%.3f,maxrate:%.3f,meancrossrate:%.3f,invailrate:%.3f'%(minrate/num,maxrate/num,meancrossrate/num,invailrate/num))
                pbar.write('aveminmse:%.3f avemaxmse:%.3f'%(aveminmse/num,avemaxmse/num))
                pbar.write('cover_rate:%.3f invaliave:%.3f'%(cover_rate/num,invaliave/invailrate))
            '''
            '''
            if pbar.n % 10000 == 0:
                if log_dir=='real':
                    writer.add_scalar('example_1',y1[0,0].item(),pbar.n)
                    writer.add_scalar('example_2',y1[0,1].item(),pbar.n)
                    writer.add_scalar('example_3',y1[0,2].item(),pbar.n)
                else:
                    writer.add_scalar('example_1',output[0,0].item(),pbar.n)
                    writer.add_scalar('example_2',output[0,1].item(),pbar.n)
                    writer.add_scalar('example_3',output[0,2].item(),pbar.n)
                    
    avemin/=num
    avemax/=num
    avemean/=num
    aveminloss/=num
    avemaxloss/=num
    aveminmse/=num
    avemaxmse/=num
    meancrossrate/=num
    invailrate/=num
    minrate/=num
    maxrate/=num
    with open('eval_log1/'+log_dir+'/'+model_name+'_eval.txt','w') as f:
        try:
            print('-------------------'+model_name+'------'+str(num))
            print('avemin:%.3f avemax:%.3f avemean:%.3f aveminloss:%.3f avemaxloss:%.3f '%(avemin,avemax,avemean,aveminloss,avemaxloss))
            print('minrate:%.3f,maxrate:%.3f,meancrossrate:%.3f,invailrate:%.3f'%(minrate,maxrate,meancrossrate,invailrate))
            print('aveminmse:%.3f avemaxmse:%.3f'%(aveminmse,avemaxmse))
            print('cover_rate:%.3f '%(cover_rate/num))
            temp=sys.stdout
            sys.stdout=f
            print('-------------------'+model_name+'------'+str(num))
            print('avemin:%.3f avemax:%.3f avemean:%.3f aveminloss:%.3f avemaxloss:%.3f '%(avemin,avemax,avemean,aveminloss,avemaxloss))
            print('minrate:%.3f,maxrate:%.3f,meancrossrate:%.3f,invailrate:%.3f'%(minrate,maxrate,meancrossrate,invailrate))
            print('aveminmse:%.3f avemaxmse:%.3f'%(aveminmse,avemaxmse))
            print('cover_rate:%.3f '%(cover_rate/num))
            sys.stdout=temp
        except:
            pass
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
    ).cuda(),'p7']
]

if __name__=='__main__':
    dloader._init()
    for i in range(len(eval_set)):
        eval(eval_set[i][0],eval_set[i][1],eval_set[i][2],eval_set[i][3])