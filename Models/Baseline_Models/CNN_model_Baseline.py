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
        targets=None,  # 
        sequence_length=sequence_length,
        sequence_stride=1,
        sampling_rate=1,
        batch_size=None,  # 
        shuffle=False
    )#(109,30)

    num_sequences = len(input_sequences)

# 
    label_sequences = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=labels,
        targets=None,  # 
        sequence_length=sequence_length,
        sequence_stride=1,
        sampling_rate=1,
        batch_size=None,  # 
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

#define model
class CNN1d(nn.Module):
    def __init__(self, n_features=1,seq_len=30,kernel_size=3):
        super(CNN1d,self).__init__()
        self.seq_len=seq_len
        self.conv1=nn.Conv1d(in_channels=n_features,out_channels=32,kernel_size=kernel_size)
        self.conv2=nn.Conv1d(in_channels=32,out_channels=16,kernel_size=kernel_size)
        self.pool=nn.AdaptiveAvgPool1d(self.seq_len)
        self.fc1=nn.Linear(16*self.seq_len,64)
        self.fc2=nn.Linear(64,30)
        self.dropout=nn.Dropout(0.2)
    def forward(self, x):
        batch_size, _, _ = x.size()  # x: [batch_size, 30, n_features]

        # Step 1: permute to match Conv1d input format
        x = x.permute(0, 2, 1)  
        # x: [batch_size, n_features, 30]

        # Step 2: first convolution
        out = nn.functional.relu(self.conv1(x))  
        # out: [batch_size, 8, 28]  
        

        # Step 3: second convolution
        out = nn.functional.relu(self.conv2(out))  
        # out: [batch_size, 16, 26]

        # Step 4: adaptive average pooling → 时间步变为 self.seq_len=30
        out = self.pool(out)  
        # out: [batch_size, 16, 30]

        # Step 5: dropout
        out = self.dropout(out)  
        # out: [batch_size, 16, 30]

        # Step 6: flatten for linear layer
        out = out.view(batch_size, -1)  
        # out: [batch_size, 16 * 30] = [batch_size, 480]

        # Step 7: fully connected layer 1
        out = nn.functional.relu(self.fc1(out))  
        # out: [batch_size, 64]

        # Step 8: dropout
        out = self.dropout(out)  
        # out: [batch_size, 64]

        # Step 9: fully connected layer 2
        out = self.fc2(out)  
        # out: [batch_size, 30]

        # Step 10: reshape to match (batch_size, time_steps, 1 output per step)
        return out.view(batch_size, 30, 1)  
        # Final shape: [batch_size, 30, 1]

k=[109,109,73]


def train_cnn_final_model(lr=0.001, epochs=300, early_stop_threshold=0.0019):
    print("\n" * 2)
    print("Training final CNN model with sliding window")
    print("-" * 50)

    model = CNN1d()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        epoch_y_true = []
        epoch_y_pred = []

        for battery_id in range(len(k)):  # B06, B07, B18
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
                input_window = x_train_b[idx].unsqueeze(0)  # [1, 30, 1]
                target_window = y_train_b[idx].unsqueeze(0)  # [1, 30, 1]

                y_pred = model(input_window)
                loss = criterion(y_pred, target_window)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())
                epoch_y_true.append(target_window.detach().cpu().numpy().reshape(-1))
                epoch_y_pred.append(y_pred.detach().cpu().numpy().reshape(-1))

        # 
        y_true_all = np.concatenate(epoch_y_true)
        y_pred_all = np.concatenate(epoch_y_pred)
        mae = mean_absolute_error(y_true_all, y_pred_all)
        rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
        r2 = r2_score(y_true_all, y_pred_all)
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch + 1:3d} | Loss={avg_loss:.4f} | MAE={mae:.4f} | RMSE={rmse:.4f} | R²={r2:.4f}")

        if avg_loss <= early_stop_threshold:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    
    torch.save(model.state_dict(), 'CNN1d_model_Baseline.pth')
    print("Final CNN model saved to 'CNN1d_model_Baseline.pth'")
#train_cnn_final_model()

def test_cnn_model(seq_len=30, model_path="CNN1d_model_Baseline.pth"):
    
    path_B05 = r"C:\Users\zheng\Desktop\RUL_Battery\Data for model\HI_B05.csv"
    df_raw = pd.read_csv(path_B05)
    label = df_raw.iloc[:, 0].values  # shape: [168]

    
    first_window = label[:seq_len]
    input_window = torch.tensor(first_window.reshape(1, seq_len, 1), dtype=torch.float32)  # [1, 30, 1]

    # load CNN 
    model = CNN1d(n_features=1, seq_len=30)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    sliding_prediction = []
    with torch.no_grad():
        for idx in range(500):
            y_pre = model(input_window)  # [1, 30, 1]
            y_pre_last = y_pre.clone()   # 

            if idx == 0:
                
                sliding_prediction.append(y_pre.squeeze().tolist())  # [30]
            else:
                
                sliding_prediction.append(y_pre.squeeze().tolist()[-1])

            # Build next input
            y_add = y_pre[:, 0:1, :]  # [1, 1, 1]
            input_window = torch.cat([input_window[:, 1:, :], y_add], dim=1)  

            
            if len(sliding_prediction) == 109:
                break

        # concatenate
        sliding_prediction = sliding_prediction[0] + sliding_prediction[1:]
        total = first_window.tolist() + sliding_prediction

    return sliding_prediction, total, label


 
sliding_prediction_B05_CNN, total_B05_CNN, label_B05 = test_cnn_model()

y_true = np.array(label_B05[30:], dtype=float)  
y_pred = np.array(sliding_prediction_B05_CNN, dtype=float)  

# MAE
mae = np.mean(np.abs(y_true - y_pred))

# R^2
ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

print(f"MAE = {mae:.6f}")
print(f"R^2 = {r2:.6f}")
#df = pd.DataFrame({
#    'Label_B05': label_B05,
#    'Total_B05_CNN': total_B05_CNN
#})


#df.to_csv('B05_results.csv', index=False)

#
#plt.figure(figsize=(10, 5))
#plt.plot(range(168), label_B05, label="True HI", color="b")
#plt.plot(range(168), total_B05_CNN, label="Predicted HI (CNN)", color="orange", linestyle='--')
#plt.axvline(30, color='gray', linestyle='--', label='Prediction Start')
#plt.xlabel("Cycle Index")
#plt.ylabel("Health Indicator")
#plt.title("Battery B05: True vs Predicted (CNN 30→30 Sliding)")
#plt.legend()
#plt.grid(False)
#plt.tight_layout()
#plt.show()
