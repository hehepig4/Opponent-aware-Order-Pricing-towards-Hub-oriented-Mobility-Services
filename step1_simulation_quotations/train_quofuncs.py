import configure
import data_vision
from tqdm import tqdm
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import func1,func2,func3,func4,func5,func6,func7,func8,func9
if not configure.if_train:
    exit()
data_vision._init()

data_vision.set_value('handler',data_vision.datahandler())

funcs=[func1]
rmse_list=[]
import time
i=0
for func in funcs:
    print("💯training: start training func"+str(i+1))
    time_start=time.time()
    try:
        rmse_list.append(func.functrain(configure.funcs_data_num[i]))
    except:
        print("🤡error: failed in training func",str(i+1))
        rmse_list.append(-1)
        continue
    print("🤣success: func",str(i+1),' finished in ',time.time()-time_start,'s with RMSE:',rmse_list[-1])
    i+=1
import matplotlib.pyplot as plt
plt.plot(range(len(rmse_list)),rmse_list)
plt.savefig('rmse_list.png')