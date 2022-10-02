import dataloader
import configure
import torch
import torch.nn as nn
import predic_model as pmodel

import torch.utils.tensorboard as tb
writer = tb.SummaryWriter('logs3/',comment='model_with_2_loss_2')

def quantile_loss(predic,y,quantile=[0.2,0.5,0.8]):
    loss=[]
    for i in range(len(quantile)):
        error=predic[:,i]-y[:,i]
        loss.append(torch.maximum((quantile[i]-1)*error,quantile[i]*error).mean())
    res=torch.stack(loss)
    return res.sum()

def elite_loss(predic,y,base):
    #predic:real
    #y:rank_target
    #base:all_reals
    minus=[]
    y=torch.tensor(y,dtype=torch.float32).cuda()
    for p in range(predic.size(dim=1)):
        minus.append([])
        
    for p in range(predic.size(dim=1)):
        for i in range(base.size(dim=1)):
            loss=base[:,i]-predic[:,p]
            minus[p].append(torch.tanh(loss))
            
    if y.size(dim=0)==predic.size(dim=0):
        for b in range(y.size(dim=1)):
            y[b]=base.size(dim=1)+1-2*y[b]
    else:
        ass=base.size(dim=1)+1-2*y
        all=[]
        for i in range(predic.size(dim=0)):
            all.append(ass)
        y=torch.stack(all,dim=0)
        
    losses=[]
     
    for p in range(predic.size(dim=1)):
        losses.append(torch.stack(minus[p]).sum(dim=0))
    res=torch.stack(losses,dim=1)
    constant=(torch.div(torch.abs(base[:,-1]-base[:,0]),base[:,5]))
    loss=0.5*torch.mul(torch.mul(torch.sub(res,y),torch.sub(res,y)).sum(dim=1),constant).mean()
    return loss

def check_point(model,optimizer,epochs,scheduler,loss,vali_loss,save_path='checkpoint/predic_model_10gpu_epochs_'):
    torch.save({
        'epochs': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()}, save_path+str(epochs)+'.pth')
    writer.add_scalar('loss', loss, epochs)
    writer.add_scalar('vali_loss', vali_loss, epochs)
    writer.add_scalar('difference', loss-vali_loss, epochs)
    
    
from torch.optim import Adam,RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.nn import MSELoss
import tqdm


def train(epochs=-1,save_path='checkpoint/predic_model_10gpu_epochs_'):
    
    
    model=pmodel.predic_model_2(
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
    ).cuda()
    model=torch.nn.DataParallel(model,device_ids=[0,1])
    optimizer=Adam(model.parameters(),lr=0.001)
    scheduler=ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=5)
    
    if epochs!=-1:
        loa=torch.load(save_path+str(epochs)+'.pth')
        if loa==None:
            print('load model failed')
            return
        model.load_state_dict(loa['model_state_dict'])
        optimizer.load_state_dict(loa['optimizer_state_dict'])
        scheduler.load_state_dict(loa['scheduler_state_dict'])
    
    for epoch in range(epochs+1,configure.epochs):
        dataset=dataloader.streamingdataset4(
            start=0,
            end=50000000
        )
    
        train_set,vali_set=torch.utils.data.random_split(dataset, [int(len(dataset)*0.8),len(dataset)-int(len(dataset)*0.8)])
        train_loader=DataLoader(train_set,batch_size=8,shuffle=True,num_workers=8,pin_memory=True)
        vali_loader=DataLoader(vali_set,batch_size=8,shuffle=False,num_workers=8,pin_memory=True)
        train_loss=0
        vali_loss=0
        print('epoch:',epoch)
        loss_fn=elite_loss
        loss_fn2=MSELoss()
        quantiles=torch.tensor(configure.percentiles,dtype=torch.float32).cuda()
        quantiles=quantiles*10
        with tqdm.tqdm(train_loader,desc='train') as pbar:
            for (x1,x2,y1,y2) in train_loader:
                optimizer.zero_grad()
                x1,x2,y1,y2=x1.cuda(),x2.cuda(),y1.cuda(),y2.cuda()
                output=model(x1,x2)
                loss=loss_fn(output,[1,5,10],y2)+loss_fn2(output,y1)
                #loss=loss_fn(output,y1)
                loss.backward()
                optimizer.step()
                if pbar.n % 20 == 0:
                    pbar.write('example: '+str(round(output[0][0].item(),3))+' '+str(round(output[0][1].item(),3))+' '+ str(round(output[0][2].item(),3))+' | '+str(round(y1[0][0].item(),3))+' '+str(round(y1[0][1].item(),3))+' '+ str(round(y1[0][2].item(),3))+' | '+str(loss.item()))
                    writer.add_scalar('exmple_loss', loss.item(),epoch*len(train_loader) + pbar.n)
                    writer.flush()
                train_loss+=loss.item()
                pbar.update(1)
                
        train_loss/=len(train_loader)
        
        with torch.no_grad():
            for (x1,x2,y1,y2) in tqdm.tqdm(vali_loader,desc='vali'):
                x1,x2,y1,y2=x1.cuda(),x2.cuda(),y1.cuda(),y2.cuda()
                output=model(x1,x2)
                loss=loss_fn(output,[1,5,10],y2)+loss_fn2(output,y1)
                #loss=loss_fn(output,y1)
                vali_loss+=loss.item()
             
        vali_loss/=len(vali_loader)
        scheduler.state_dict
        scheduler.step(vali_loss)
        check_point(model,optimizer,epoch,scheduler,train_loss,vali_loss,save_path=save_path)
        try:
            print('epoch ',epoch,' train_loss ',train_loss,' vali_loss ',vali_loss,file=open(save_path+'sum.txt','a'))
        except:
            print('epoch ',epoch,' train_loss ',train_loss,' vali_loss ',vali_loss)
if __name__=='__main__':
    dataloader._init()
    train(27)                                                                                       
