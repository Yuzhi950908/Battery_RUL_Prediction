import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import tensorflow as tf
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Input Data
#################################################################################################################################################################################
df={}
battery_index=["B06","B07","B18"]
for i in range(len(battery_index)):
    file_name = f"HI_{battery_index[i]}.csv"
    file_path = os.path.join(r"C:\Users\zheng\Desktop\RUL_Battery\Data for model",file_name)
    df_temp = pd.read_csv(file_path)
    df[battery_index[i]]=df_temp.iloc[:,0]
#df=pd.DataFrame(df)
#df.to_csv("df.csv",index=False)

#########################################################################################################################################################################################



#First step: build the train instance
###################################################################################################################################################################################################
past=30
future=30
windowsize=past+future
sequence_length = 30
dataset={}
for key,value in df.items():
    input=df[key][:-future]#len:138
    labels =df[key][past:]#len:138

    input_sequences = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=input,
        targets=None, 
        sequence_length=sequence_length,
        sequence_stride=1,
        sampling_rate=1,
        batch_size=None,  
        shuffle=False
    )#(109,30)

    num_sequences = len(input_sequences)

#
    label_sequences = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=labels,
        targets=None,  
        sequence_length=sequence_length,
        sequence_stride=1,
        sampling_rate=1,
        batch_size=None,  
        shuffle=False
    )#(109,30)

    x_train = np.array(list(input_sequences.as_numpy_iterator()))#(109,30)
    y_train = np.array(list(label_sequences.as_numpy_iterator()))#(109,30)
    x_train= np.expand_dims(x_train,axis=-1)#(109,30,1)
    y_train= np.expand_dims(y_train,axis=-1)#(109,30,1)
    dataset[key]={"input":x_train,"label":y_train}



class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train=[]
for key,value in dataset.items():
    train_temp=MyDataset(value["input"],value["label"]) 
    train.append(train_temp)


x_train=[]
label_train=[]

for list in train:
    x_train.append(list.x)
    label_train.append(list.y)


x_train_all=torch.cat(x_train,0) #torch.Size[291,30,1]
y_train_all=torch.cat(label_train,0) #torch.Size[291,30,1]



x_train_all=x_train_all.permute(0,2,1)
x_train_all=x_train_all.repeat(1,30,1)
#Now x (291，30，30)
# y (291，30，1)


#build K-folder
k=[109,109,73]



#define model
"""
Please Note! The construction of the Transformer model was based on 
Zhou’s open-source code repository, which can be found at the following address:
https://github.com/XiuzeZhou/RUL
"""
class PositionalEncoding(nn.Module):
    def __init__(self, feature_len,feature_size,dropout=0):
        """
        feature_len = window_size
        feature_size = num of HIs
        """
        super(PositionalEncoding,self).__init__()
        pe=torch.zeros(feature_len,feature_size)# -> 
        position=torch.arange(0,feature_len,dtype=torch.float).unsqueeze(1)#boardcast。 [[0],[1],[2]....]
        #Attension is all ur need ->sin cos 
        #
        div_term=torch.exp(torch.arange(0,feature_size,2).float()*(-math.log(10000.0) / feature_size))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self,x):
        x=x+self.pe
        return x

class Transformer(nn.Module):
    def __init__(self,inputsize,feature_num,num_layers,nhead):
        super(Transformer,self).__init__()
        self.hidden_dim=64

        if feature_num==1:
            self.pos=PositionalEncoding(feature_len=inputsize,feature_size=inputsize)
            encoder_layers= nn.TransformerEncoderLayer(
                d_model=inputsize,
                nhead=nhead,
                dim_feedforward=self.hidden_dim,
                dropout=0.2,
                batch_first=True
            )
        self.cell=nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.linear = nn.Linear(inputsize,1)
       

    def forward(self, x): 
        batch_size, _ , feature_num = x.shape
        out = self.pos(x)
        out = self.cell(out)              # sigle feature: (batch_size, feature_num, auto_hidden) or multi-features: (batch_size, auto_hidden, feature_num) # (batch_size, feature_num*inputsize)
        out = self.linear(out)            
        return out

#train model
print("Loading 3-Folder training and validation")
print("-" * 50)

epochs=100
loss_train={}
loss_val={}





#Train final_model
print("\n" * 2)
print("Loading final training with sliding window")
print("-" * 50)

def train_final(lr=0.0001,epochs=300,early_stop_threshold=0.0012):
    model=Transformer(inputsize=30,feature_num=1,num_layers=2,nhead=2)
    criterion=nn.MSELoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_losses=[]
        epoch_y_true=[]
        epoch_y_pred=[]

        for battery_id in range(len(k)): # 0=B06, 1=B07, 2=B18
            if battery_id == 0:
                x_train_b = x_train_all[0:k[0]]
                y_train_b = y_train_all[0:k[0]]
            elif battery_id == 1:
                x_train_b = x_train_all[k[0]:k[0] + k[1]]
                y_train_b = y_train_all[k[0]:k[0] + k[1]]
            else:
                x_train_b = x_train_all[k[0] + k[1]:]
                y_train_b = y_train_all[k[0] + k[1]:]

            for idx in range(k[battery_id]):
                input_window=x_train_b[idx].unsqueeze(0) 
                y_pre=model(input_window)
                loss= criterion(y_pre,y_train_b[idx].unsqueeze(0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())
                epoch_y_true.append(y_train_b[idx].detach().cpu().numpy().reshape(-1))
                epoch_y_pred.append(y_pre.detach().cpu().numpy().reshape(-1))

        y_true_all = np.concatenate(epoch_y_true)
        y_pred_all = np.concatenate(epoch_y_pred)
        mae = mean_absolute_error(y_true_all, y_pred_all)
        rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
        r2 = r2_score(y_true_all, y_pred_all)
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1:3d} | Loss={avg_loss:.4f} | MAE={mae:.4f} | RMSE={rmse:.4f} | R²={r2:.4f}")
        if avg_loss <= early_stop_threshold:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    #torch.save(model.state_dict(), 'Transformer_baseline.pth')
    #print("Final model saved to 'Transformer_baseline.pth'")        
#train_final()



path_B05=(r"C:\Users\zheng\Desktop\RUL_Battery\Data for model\HI_B05.csv")
df_raw = pd.read_csv(path_B05)       
label = df_raw.iloc[:,0].values      

first_window = label[:30]            
Tensor_test = torch.tensor(first_window.reshape(1, 30, 1), dtype=torch.float32)


def test():
    model = Transformer(inputsize=30,feature_num=1,num_layers=2,nhead=2)
    model.load_state_dict(torch.load("Transformer_baseline.pth"))
    model.eval()
    with torch.no_grad():
        sliding_prediction=[]
        for idx in range(500):
            if idx == 0:
                input_window = Tensor_test
            else:
                y_add = y_pre_last[:, 0:1, :]  # [1,1,1]
                input_window = torch.cat([input_window[:, 1:, :], y_add], dim=1)
            
            y_pre = model(input_window)
            y_pre_last = y_pre.clone()

            if idx == 0:
                sliding_prediction.append(y_pre.squeeze().tolist())
            else:
                sliding_prediction.append(y_pre.squeeze().tolist()[-1])#

            if len(sliding_prediction)== 109: 
                break
        
        sliding_prediction=sliding_prediction[0]+sliding_prediction[1:]
        total=first_window.tolist()+sliding_prediction
    return sliding_prediction,total

sliding_prediction_B05,total_B05=test()
y_true = np.array(label[30:], dtype=float) 
y_pred = np.array(sliding_prediction_B05, dtype=float)  

# MAE
mae = np.mean(np.abs(y_true - y_pred))

# R^2
ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

print(f"MAE = {mae:.6f}")
print(f"R^2 = {r2:.6f}")
#df = pd.read_csv('B05_results.csv')
#df['Total_B05_Transformer'] = total_B05
#df.to_csv('B05_results.csv', index=False)
































































r"""
plt.figure(figsize=(10, 5))
plt.plot(range(168), label, label="True HI", color="b")
plt.plot(range(168), total_B05, label="Predicted HI", color="orange", linestyle='--')
plt.axvline(30, color='gray', linestyle='--', label='Prediction Start')
plt.xlabel("Cycle Index")
plt.ylabel("Health Indicator")
plt.title("Battery B05: True vs Predicted (30→30 Transformer)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



print("ignore under infos, its just for me dubug model and compare")
path_B06 = r"C:\Users\zheng\Desktop\RUL_Battery\Data for model\HI_B06.csv"
df_b06 = pd.read_csv(path_B06)
label_b06 = df_b06.iloc[:, 0].values
first_window_b06 = label_b06[:30]
Tensor_test_b06 = torch.tensor(first_window_b06.reshape(1, 30, 1), dtype=torch.float32)


def test_on_input(initial_input):
    model = Transformer(inputsize=30,feature_num=1,num_layers=2,nhead=2)
    model.load_state_dict(torch.load("Transformer_baseline.pth"))
    model.eval()
    with torch.no_grad():
        sliding_prediction = []
        for idx in range(500):
            if idx == 0:
                input_window = initial_input
            else:
                y_add = y_pre_last[:, 0:1, :]
                input_window = torch.cat([input_window[:, 1:, :], y_add], dim=1)

            y_pre = model(input_window)
            y_pre_last = y_pre.clone()

            if idx == 0:
                sliding_prediction.append(y_pre.squeeze().tolist())
            else:
                sliding_prediction.append(y_pre.squeeze().tolist()[-1])

            if len(sliding_prediction) == 109:
                break

        sliding_prediction = sliding_prediction[0] + sliding_prediction[1:]
        total = initial_input.squeeze().tolist() + sliding_prediction
    return total

pred_b06 = test_on_input(Tensor_test_b06)
plt.figure(figsize=(10, 5))
plt.plot(range(168), label_b06, label="True HI (B06)", color="b")
plt.plot(range(168), pred_b06, label="Predicted HI (B06)", color="orange", linestyle='--')
plt.axvline(30, color='gray', linestyle='--', label='Prediction Start')
plt.xlabel("Cycle Index")
plt.ylabel("Health Indicator")
plt.title("Sanity Check on B06: True vs Predicted (30→30 Transformer)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
"""