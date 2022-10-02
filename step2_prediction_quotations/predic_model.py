import torch.nn as nn
import torch
import convlstm
class predic_model(nn.Module):
    def __init__(self,kernel_size,input_dim,hidden_dim,convlstm_num_layers,
                 linear_dim,
                 seq_len,input_width,input_height,output_dim
                 ,dropout=0.3):
        super(predic_model, self).__init__()
        self.conv_lstm = convlstm.ConvLSTM(input_dim=input_dim,
                                           kernel_size=kernel_size,
                                           hidden_dim=hidden_dim,
                                           num_layers=convlstm_num_layers,
                                           return_all_layers=False,
                                           batch_first=True)
        #linear for float vars
        self.linear1=nn.Linear(hidden_dim*input_width*input_height,)
        self.Flatten=nn.Flatten()
        self.linears=[nn.Linear(hidden_dim*input_width*input_height,linear_dim[0])]
        
        for i in range(len(linear_dim)-1):
            self.linears.append(nn.ReLU())
            self.linears.append(nn.Linear(linear_dim[i],linear_dim[i+1]))
            self.linears.append(nn.Dropout(dropout))
        self.linears.append(nn.ReLU())
        self.linears.append(nn.Linear(linear_dim[-1],output_dim))
        self.linears=nn.ModuleList(self.linears)
        self.linears=nn.Sequential(*self.linears)
        
    def forward(self, x):
        #input([batch_size,channal,height,width,seq_len],[batch_size,floatvar_num,seq_len])
        x,_=self.conv_lstm(x)
        #x=x[:,-1,:,:,:]
        x=self.Flatten(x)
        x=self.linears(x)
        
        return x
    
class predic_model_2(nn.Module):
    def __init__(self,
                 kernel_size,input_dim,hidden_dim,input_height,input_width,seq_len,
                 linear_dim,output_dim,convlstm_linear_size,dropout=0.3,var_num=5):
        super(predic_model_2, self).__init__()
        self.convlstm=convlstm.ConvLSTM(
            input_dim=input_dim,
            kernel_size=kernel_size,
            hidden_dim=hidden_dim,
            num_layers=1,
            return_all_layers=False,
            batch_first=True
        )
        
        self.Faltten=nn.Flatten()
        self.linear1=[nn.Linear(hidden_dim*input_width*input_height,convlstm_linear_size)]
        self.linear2=[nn.Linear(var_num,10)]
        self.linear1.append(nn.ReLU())
        self.linear2.append(nn.ReLU())
        self.linear1=nn.ModuleList(self.linear1)
        self.linear2=nn.ModuleList(self.linear2)
        self.linear1=nn.Sequential(*self.linear1)
        self.linear2=nn.Sequential(*self.linear2)
        self.linears=[nn.Linear(10+convlstm_linear_size,linear_dim[0])]
        
        for i in range(len(linear_dim)-1):
            self.linears.append(nn.ReLU())
            self.linears.append(nn.Linear(linear_dim[i],linear_dim[i+1]))
            self.linears.append(nn.Dropout(dropout))
        self.linears.append(nn.ReLU())
        self.linears.append(nn.Linear(linear_dim[-1],output_dim))
        self.linears.append(nn.ReLU())
        self.linears=nn.ModuleList(self.linears)
        self.linears=nn.Sequential(*self.linears)
        
        
    def forward(self, x1,x2):
        x1,_=self.convlstm(x1)
        x1=self.Faltten(x1)
        x1=self.linear1(x1)
        x2=self.linear2(x2)
        x=torch.cat((x1,x2),dim=1)
        x=self.linears(x)
        return x
    
class predic_model_3(nn.Module):
    def __init__(self,
                 kernel_size,input_dim,hidden_dim,input_height,input_width,seq_len,
                 linear_dim,output_dim,convlstm_linear_size,dropout=0.3,var_num=5):
        super(predic_model_3, self).__init__()
        self.convlstm=convlstm.ConvLSTM(
            input_dim=input_dim,
            kernel_size=kernel_size,
            hidden_dim=hidden_dim,
            num_layers=1,
            return_all_layers=False,
            batch_first=True
        )
        
        self.Faltten=nn.Flatten()
        self.linear1=[nn.Linear(hidden_dim*input_width*input_height,convlstm_linear_size)]
        self.linear2=[nn.Linear(var_num,10)]
        self.linear1.append(nn.ReLU())
        self.linear2.append(nn.ReLU())
        self.linear1=nn.ModuleList(self.linear1)
        self.linear2=nn.ModuleList(self.linear2)
        self.linear1=nn.Sequential(*self.linear1)
        self.linear2=nn.Sequential(*self.linear2)
        self.linears=[nn.Linear(10+convlstm_linear_size,linear_dim[0])]
        
        for i in range(len(linear_dim)-1):
            self.linears.append(nn.ReLU())
            self.linears.append(nn.Linear(linear_dim[i],linear_dim[i+1]))
            self.linears.append(nn.Dropout(dropout))
        self.linears.append(nn.ReLU())
        self.linears.append(nn.Linear(linear_dim[-1],output_dim))
        self.linears.append(nn.ReLU())
        self.linears=nn.ModuleList(self.linears)
        self.linears=nn.Sequential(*self.linears)
        
        
    def forward(self, x1,x2):
        x1,_=self.convlstm(x1)
        x1=self.Faltten(x1)
        x1=self.linear1(x1)
        x2=self.linear2(x2)
        x=torch.cat((x1,x2),dim=1)
        x=self.linears(x)
        return x