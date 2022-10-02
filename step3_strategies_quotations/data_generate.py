import torch 
import pandas as pd
import numpy as np
import configure
class QuanQPred(torch.nn.Module):
    def __init__(self,linear_dims,conv_dims,kernel_size,final_dims,quans,input_height=81,input_width=131,dropout=0.3):
        super(QuanQPred,self).__init__()
        self.linears=[]
        for i in range(len(linear_dims)-1):
            self.linears.append(torch.nn.Linear(linear_dims[i],linear_dims[i+1]))
            self.linears.append(torch.nn.Dropout(dropout))
            self.linears.append(torch.nn.ReLU())
        assert len(conv_dims)==len(kernel_size)
        self.convs=[]
        for i in range(len(conv_dims)-2):
            self.convs.append(
                torch.nn.Conv2d(conv_dims[i],conv_dims[i+1],kernel_size=kernel_size[i])
            )
            self.convs.append(torch.nn.BatchNorm2d(conv_dims[i]))
            self.convs.append(torch.nn.ReLU())
        self.convs.append(torch.nn.Flatten())
        self.linears=torch.nn.ModuleList(self.linears)
        self.convs=torch.nn.ModuleList(self.convs)
        self.linears=torch.nn.Sequential(*self.linears)
        self.convs=torch.nn.Sequential(*self.convs)

        
        _conv_size=self.convs(torch.zeros((1,3,input_height,input_width))).shape[-1]
        self.final_linears=[torch.nn.Linear(_conv_size+linear_dims[-1],final_dims[0]),
                           torch.nn.Dropout(dropout),
                           torch.nn.ReLU()]
        for i in range(len(final_dims)-1):
            self.final_linears.append(torch.nn.Linear(final_dims[i],final_dims[i+1]))
            self.final_linears.append(torch.nn.Dropout(dropout))
            self.final_linears.append(torch.nn.ReLU())
        self.final_linears.append(torch.nn.Linear(final_dims[-1],quans))
        self.final_linears=torch.nn.ModuleList(self.final_linears)
        self.final_linears=torch.nn.Sequential(*self.final_linears)

    def forward(self,linear_fea,conv_fea):
        x1,x2=self.linears(linear_fea),self.convs(conv_fea).float()
        x=torch.cat((x1,x2),dim=1)
        x=self.final_linears(x)
        return x
    def getloss(self,pred,target):
        return torch.nn.functional.mse_loss(pred,target)
    def save(self,step,path='QuanQPred/'):
        torch.save(self.state_dict(),path+'QuanQPred_'+str(step)+'.pth')
    def load(self,path):
        self.load_state_dict(torch.load(path))
model=QuanQPred(
        linear_dims=[2,8,16],
        conv_dims=[3,5],
        kernel_size=[5,5],
        final_dims=[32],
        quans=8).to('cuda')
model.load('QuanQPred/QuanQPred_99.pth')
reqdata=pd.read_hdf('test_data/new_testdata.h5')
time_min=reqdata['abs_minutes'].min()
time_max=reqdata['abs_end_minutes'].max()
zdata=torch.load('backwardten.pt')
reqdata['pmin']=reqdata['weighted_pinball_loss_Rank_loss_min']
reqdata['pmax']=reqdata['weighted_pinball_loss_Rank_loss_max']
reqdata['pmid']=reqdata['weighted_pinball_loss_Rank_loss_mean']
class genQDataset(torch.utils.data.Dataset):
    def __init__(self,time_min,time_max):
        self.zdata=zdata
        self.length=time_max-time_min+1
        self.time_min=time_min
        self.time_max=time_max
    def __len__(self):
        return self.length
    def __getitem__(self,index):
        weekday=(index+self.time_min)//(24*60)
        weekday=weekday%7
        minutes_in_day=(index+self.time_min)%(24*60)
        feas=[
            weekday,
            minutes_in_day
        ]
        feas=torch.tensor(feas).float()
        zdata=self.zdata[weekday*1440+minutes_in_day]
        return zdata,feas
handle=[]
time_rec=[]
@torch.no_grad()
def generate():
    dataset=genQDataset(time_min,time_max)
    data_loader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
    for i,(zdata,feas) in enumerate(data_loader):
        time_rec.append(feas)
        zdata=zdata.to('cuda')
        feas=feas.to('cuda')
        pred=model(feas,zdata)
        handle.append(pred.cpu())
    Qhandle=torch.stack(handle,dim=0)
    timerec=torch.stack(time_rec,dim=0)
    torch.save(Qhandle,'test_data/Qhandle.pt')
    torch.save(timerec,'test_data/timerec.pt')
@torch.no_grad()
def gernerate_q():
    history_time_min=time_min%(1440*7)
    history_time_max=time_max%(1440*7)
    torch.save(zdata[history_time_min:history_time_max],'test_data/Zdata.pt')
    reqdata['quantities']=0
    def get_block(latitudes,longitudes):
        return (int(np.round((latitudes-configure.min_la)/configure.step_dis)),
                int(np.round((longitudes-configure.min_lo)/configure.step_dis))) 

    def get_quan(start_block,end_block,start_time,end_time,q):
        quan=zdata[end_time][0][end_block]*np.power(0.99,end_time-start_time)+q-zdata[start_time][0][start_block]
        return quan.item()
    for i in range(reqdata.shape[0]):
        block_start=get_block(reqdata['pickup_latitude'][i],reqdata['pickup_longitude'][i])
        block_end=get_block(reqdata['dropoff_latitude'][i],reqdata['dropoff_longitude'][i])
        q=reqdata['pmid'][i]
        start_time=int(reqdata['abs_minutes'][i])
        end_time=int(reqdata['abs_end_minutes'][i])
        quan=get_quan(block_start,block_end,start_time,end_time,q)
        reqdata['quantities'].at[i]=quan
    reqdata.to_hdf('test_data/reqdata.h5','data')
if __name__=='__main__':
    generate()
    gernerate_q()
#    reqdata.to_csv('test_data/new_testdata.csv')