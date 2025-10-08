import pywt
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat
import warnings
warnings.filterwarnings("ignore")
from scipy import stats
# funcitions
def madev(d,axis=None):
    return np.mean(np.absolute(d- np.mean(d,axis)),axis)
def DWT_noisy_smooth(raw_HI, wavelet="db4", level=5):
    coeffs = pywt.wavedec(raw_HI, wavelet, level=level)
    coeffs[1:] = [np.zeros_like(detail) for detail in coeffs[1:]]
    trend_only = pywt.waverec(coeffs, wavelet) 
    return trend_only[:len(raw_HI)]

    
def min_max_normalization(smooth_HI):
    min_val = np.min(smooth_HI)
    max_val = np.max(smooth_HI)
    HI=[(x - min_val) / (max_val - min_val) for x in smooth_HI]
    return HI






#######################################################################################################
#HI1: capacity
#B0005
df_B0005 = pd.read_csv('Raw_Datas/capacity/B0005.csv')
raw_capacity_B0005 = df_B0005['capacity'].values
smooth_capacity_B0005=DWT_noisy_smooth(raw_capacity_B0005)
capacity_B0005= min_max_normalization(smooth_capacity_B0005)
capacity_B05=pd.DataFrame(capacity_B0005,columns=["capacity"])
#capacity_B0005.to_csv("capacity_B0005_.csv", index=False)
C_B05_exp=min_max_normalization(raw_capacity_B0005)
C_B05_example=raw_capacity_B0005
plt.figure(figsize=(10, 5))
plt.plot(raw_capacity_B0005, label="Raw Signal",linewidth=2,color="#00549F")
plt.plot(smooth_capacity_B0005, label="Smoothed Signal", linewidth=2)
plt.xlabel("Cycle Index")
plt.ylabel("Capacity")
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()


#B0006
df_B0006 = pd.read_csv('Raw_Datas/capacity/B0006.csv')
raw_capacity_B0006 = df_B0006['capacity'].values
smooth_capacity_B0006=DWT_noisy_smooth(raw_capacity_B0006)
capacity_B06= min_max_normalization(smooth_capacity_B0006)
C_B06_exp=min_max_normalization(raw_capacity_B0006)
#C_B06_exp=raw_capacity_B0006
#B0007
df_B0007 = pd.read_csv('Raw_Datas/capacity/B0007.csv')
raw_capacity_B0007 = df_B0007['capacity'].values
smooth_capacity_B0007=DWT_noisy_smooth(raw_capacity_B0007)
capacity_B07= min_max_normalization(smooth_capacity_B0007)
C_B07_exp=min_max_normalization(raw_capacity_B0007)
#C_B07_exp=raw_capacity_B0007
#B0018
df_B0018 = pd.read_csv('Raw_Datas/capacity/B0018.csv')
raw_capacity_B0018 = df_B0018['capacity'].values
smooth_capacity_B0018=DWT_noisy_smooth(raw_capacity_B0018)
capacity_B18= min_max_normalization(smooth_capacity_B0018)
C_B18_exp=min_max_normalization(raw_capacity_B0018)
#C_B18_exp=raw_capacity_B0018
#################################################################################################################################













###########################################################################################################################
#HI2: DVD_ETI
#B0005
data_path_B0005=os.path.join("Raw_Datas","B0005")
raw_dvd_B0005=[]
load_mat_B0005=loadmat(data_path_B0005)
raw_data_B0005=load_mat_B0005["B0005"][0][0][0][0]
data_size_B0005=raw_data_B0005.shape[0]
for rows in range(data_size_B0005):
    if raw_data_B0005[rows][0][0]=="discharge":
        DVD_startpoint_B05=raw_data_B0005[rows][3][0][0][0][0][0]
        sample_time_B05=raw_data_B0005[rows][3][0][0][-2][0] 
        sample_point_B05=next(i for i,t in enumerate(sample_time_B05) if t >=1700)
        DVD_endpoint_B05=raw_data_B0005[rows][3][0][0][0][0][sample_point_B05]
        DVD_ETI_B05=DVD_startpoint_B05-DVD_endpoint_B05
        raw_dvd_B0005.append(DVD_ETI_B05)
smooth_dvd_B0005=DWT_noisy_smooth(raw_dvd_B0005)
DVD_B05= min_max_normalization(smooth_dvd_B0005)
DVD_B05_exp= min_max_normalization(raw_dvd_B0005)
DVD_B05_example= raw_dvd_B0005


#B0006
data_path_B0006=os.path.join("Raw_Datas","B0006")
raw_dvd_B0006=[]
load_mat_B0006=loadmat(data_path_B0006)
raw_data_B0006=load_mat_B0006["B0006"][0][0][0][0]
data_size_B0006=raw_data_B0006.shape[0]
for rows in range(data_size_B0006):
    if raw_data_B0006[rows][0][0]=="discharge":
        DVD_startpoint_B06=raw_data_B0006[rows][3][0][0][0][0][0]
        sample_time_B06=raw_data_B0006[rows][3][0][0][-2][0] 
        sample_point_B06=next(i for i,t in enumerate(sample_time_B06) if t >=1700)
        DVD_endpoint_B06=raw_data_B0006[rows][3][0][0][0][0][sample_point_B06]
        DVD_ETI_B06=DVD_startpoint_B06-DVD_endpoint_B06
        raw_dvd_B0006.append(DVD_ETI_B06)
smooth_dvd_B0006=DWT_noisy_smooth(raw_dvd_B0006)
DVD_B06= min_max_normalization(smooth_dvd_B0006)
DVD_B06_exp= min_max_normalization(raw_dvd_B0006)
#DVD_B06_exp= raw_dvd_B0006


#B0007
data_path_B0007=os.path.join("Raw_Datas","B0007")
raw_dvd_B0007=[]
load_mat_B0007=loadmat(data_path_B0007)
raw_data_B0007=load_mat_B0007["B0007"][0][0][0][0]
data_size_B0007=raw_data_B0007.shape[0]
for rows in range(data_size_B0007):
    if raw_data_B0007[rows][0][0]=="discharge":
        DVD_startpoint_B07=raw_data_B0007[rows][3][0][0][0][0][0]
        sample_time_B07=raw_data_B0007[rows][3][0][0][-2][0] 
        sample_point_B07=next(i for i,t in enumerate(sample_time_B07) if t >=1700)
        DVD_endpoint_B07=raw_data_B0007[rows][3][0][0][0][0][sample_point_B07]
        DVD_ETI_B07=DVD_startpoint_B07-DVD_endpoint_B07
        raw_dvd_B0007.append(DVD_ETI_B07)
smooth_dvd_B0007=DWT_noisy_smooth(raw_dvd_B0007)
DVD_B07= min_max_normalization(smooth_dvd_B0007)
#DVD_B07_exp= min_max_normalization(raw_dvd_B0007)
DVD_B07_exp= raw_dvd_B0007
#B0018
data_path_B0018=os.path.join("Raw_Datas","B0018")
raw_dvd_B0018=[]
load_mat_B0018=loadmat(data_path_B0018)
raw_data_B0018=load_mat_B0018["B0018"][0][0][0][0]
data_size_B0018=raw_data_B0018.shape[0]
for rows in range(data_size_B0018):
    if raw_data_B0018[rows][0][0]=="discharge":
        DVD_startpoint_B18=raw_data_B0018[rows][3][0][0][0][0][0]
        sample_time_B18=raw_data_B0018[rows][3][0][0][-2][0] 
        sample_point_B18=next(i for i,t in enumerate(sample_time_B18) if t >=1700)
        DVD_endpoint_B18=raw_data_B0018[rows][3][0][0][0][0][sample_point_B18]
        DVD_ETI_B18=DVD_startpoint_B18-DVD_endpoint_B18
        raw_dvd_B0018.append(DVD_ETI_B18)
smooth_dvd_B0018=DWT_noisy_smooth(raw_dvd_B0018)
DVD_B18= min_max_normalization(smooth_dvd_B0018)
DVD_B18_exp= min_max_normalization(raw_dvd_B0018)
#DVD_B18_exp= raw_dvd_B0018
#######################################################################################################################################















#####################################################################################################
#HI3: PP_ICA
df=pd.read_csv(r"Raw_Datas\All discharge\all discharge.csv")
def cal_ica(data_frame,cycle_num):
    smooth_ICA={}
    voltage_dict={}
    for n in range(cycle_num):
        temp_index=data_frame[data_frame.iloc[:,7]==n+1].index
        time=data_frame.loc[temp_index,"Time"].tolist()
        voltage=data_frame.loc[temp_index,"Voltage_measured"].tolist()
        dU = np.diff(voltage)
        dU[dU == 0] = 1e-6
        current=data_frame.loc[temp_index,"Current_measured"].tolist()
        origin_ICA=(np.diff(time)/ dU)*np.array(current[1:])/ 3600
        temp_smooth_ICA=DWT_noisy_smooth(origin_ICA)
        smooth_ICA[n] =temp_smooth_ICA
        voltage_dict[n] = voltage
    all_data={}
    for k in smooth_ICA.keys():
        all_data[f"ICA_Cycle_{k+1}"] = pd.Series(smooth_ICA[k])
        all_data[f"Voltage_Cycle_{k+1}"] = pd.Series(voltage_dict[k][1:])
    combined_df=pd.concat(all_data,axis=1)
    return combined_df

#B0005
data_frame_B05=df[df.iloc[:,11] == "B0005"]
cycle_num_B05=data_frame_B05.iloc[:, 7].max()
combined_df_B05=cal_ica(data_frame_B05,cycle_num_B05)
#combined_df_B05.to_csv("ICA_and_Voltage_B05.csv", index=False)
"""
#draw cycle: 1，20，40，60，80，100，120，140，160
#colum index 0，38，78，118，158，198，238，278，318
y1 = combined_df_B05.iloc[:, 0]  # ICA_Cycle_1
x1 = combined_df_B05.iloc[:, 1]  # Voltage_Cycle_1

y20 = combined_df_B05.iloc[:, 38]
x20= combined_df_B05.iloc[:, 39]

y40= combined_df_B05.iloc[:, 78]
x40= combined_df_B05.iloc[:, 79]

y60= combined_df_B05.iloc[:, 118]
x60= combined_df_B05.iloc[:, 119]

y80= combined_df_B05.iloc[:, 158]
x80= combined_df_B05.iloc[:, 159]

y100= combined_df_B05.iloc[:, 198]
x100= combined_df_B05.iloc[:, 199]

y120= combined_df_B05.iloc[:, 238]
x120= combined_df_B05.iloc[:, 239]

y140= combined_df_B05.iloc[:, 278]
x140= combined_df_B05.iloc[:, 279]

y160 = combined_df_B05.iloc[:, 318]
x160 = combined_df_B05.iloc[:, 319]

cycles = {
    'Cycle 1': (x1, y1),
    'Cycle 20': (x20, y20),
    'Cycle 40': (x40, y40),
    'Cycle 60': (x60, y60),
    'Cycle 80': (x80, y80),
    'Cycle 100': (x100, y100),
    'Cycle 120': (x120, y120),
    'Cycle 140': (x140, y140),
    'Cycle 160': (x160, y160),
}

#plt.figure(figsize=(10, 6))


#for label, (x, y) in cycles.items():
#    plt.plot(x, y, label=label)

#plt.xlabel('Voltage (V)')
#plt.ylabel('ICA')
#plt.title('ICA vs Voltage for Different Cycles')
#plt.legend()
#plt.grid(True)
#plt.tight_layout()
#plt.show()
"""
# extract peak
ICA_max_B05=[]
for i in range(cycle_num_B05):
    column_data=combined_df_B05.iloc[:,i*2]
    ICA_max_B05.append(column_data.max())


#outlier detection

data_array = np.array(ICA_max_B05)
outlier_idx = np.argmax(data_array)
ICA_max_B05[outlier_idx]=(ICA_max_B05[outlier_idx-1]+ICA_max_B05[outlier_idx+1])/2



#处denoise
smooth_ICA_B0005=DWT_noisy_smooth(ICA_max_B05)
#plt.figure(figsize=(10,6))
#plt.scatter(range(cycle_num_B05),ICA_max_B05,label="with noise")
#plt.plot(smooth_ICA_B0005,color="y",label="smoothed")
#plt.xlabel("cycles")
#plt.ylabel("ICA_max")
#plt.show()
#min-max Normalization
ICA_B05= min_max_normalization(smooth_ICA_B0005)
ICA_B05_exp=min_max_normalization(ICA_max_B05)
ICA_B05_example=ICA_max_B05


#B0006
data_frame_B06=df[df.iloc[:,11] == "B0006"]
cycle_num_B06=data_frame_B06.iloc[:, 7].max()
combined_df_B06=cal_ica(data_frame_B06,cycle_num_B06)
#combined_df_B06.to_csv("ICA_and_Voltage_B06.csv", index=False)

ICA_max_B06=[]
for i in range(cycle_num_B06):
    column_data=combined_df_B06.iloc[:,i*2]
    ICA_max_B06.append(column_data.max())



#outlier
#from scipy.stats import zscore
#z_scores = zscore(ICA_max_B06)
#outliers = [val for val, z in zip(ICA_max_B06, z_scores) if abs(z) > 3]
#print("Outliers:", outliers)
#No outliers


#denoise
smooth_ICA_B0006=DWT_noisy_smooth(ICA_max_B06)
#plt.figure(figsize=(10,6))
#plt.scatter(range(cycle_num_B06),ICA_max_B06,label="with noise")
#plt.plot(smooth_ICA_B0006,color="y",label="smoothed")
#plt.xlabel("cycles")
#plt.ylabel("ICA_max")
#plt.show()
#minmax
ICA_B06= min_max_normalization(smooth_ICA_B0006)
ICA_B06_exp=min_max_normalization(ICA_max_B06)
#ICA_B06_exp=ICA_max_B06



#B0007
data_frame_B07=df[df.iloc[:,11] == "B0007"]
cycle_num_B07=data_frame_B07.iloc[:, 7].max()
combined_df_B07=cal_ica(data_frame_B07,cycle_num_B07)
#combined_df_B07.to_csv("ICA_and_Voltage_B07.csv", index=False)


ICA_max_B07=[]
for i in range(cycle_num_B07):
    column_data=combined_df_B07.iloc[:,i*2]
    ICA_max_B07.append(column_data.max())

#outlier
from scipy.stats import zscore
z_scores = zscore(ICA_max_B07)
outliers = [val for val, z in zip(ICA_max_B07, z_scores) if abs(z) > 3]
#print("Outliers:", outliers)
#one outlier
data_array = np.array(ICA_max_B07)
outlier_idx = np.argmax(data_array)
ICA_max_B07[outlier_idx]=(ICA_max_B07[outlier_idx-1]+ICA_max_B07[outlier_idx+1])/2
#denoise
smooth_ICA_B0007=DWT_noisy_smooth(ICA_max_B07)
#plt.figure(figsize=(10,6))
#plt.scatter(range(cycle_num_B07),ICA_max_B07,label="with noise")
#plt.plot(smooth_ICA_B0007,color="y",label="smoothed")
#plt.xlabel("cycles")
#plt.ylabel("ICA_max")
#plt.show()
#minmax
ICA_B07= min_max_normalization(smooth_ICA_B0007)
ICA_B07_exp=min_max_normalization(ICA_max_B07)
#ICA_B07_exp=ICA_max_B07



#B0018
data_frame_B18=df[df.iloc[:,11] == "B0018"]
cycle_num_B18=data_frame_B18.iloc[:, 7].max()
combined_df_B18=cal_ica(data_frame_B18,cycle_num_B18)
#combined_df_B18.to_csv("ICA_and_Voltage_B18.csv", index=False)

ICA_max_B18=[]
for i in range(cycle_num_B18):
    column_data=combined_df_B07.iloc[:,i*2]
    ICA_max_B18.append(column_data.max())



#outlier
from scipy.stats import zscore
z_scores = zscore(ICA_max_B18)
outliers = [val for val, z in zip(ICA_max_B18, z_scores) if abs(z) > 1]
#print("Outliers:", outliers)
outlier_idx=[i for i,t in enumerate(ICA_max_B18) if t in outliers]
outlier_idx=outlier_idx[:-2]
for idx in outlier_idx:
    ICA_max_B18[idx]=(ICA_max_B18[idx-1]+ICA_max_B18[idx+1])/2
#denoise
smooth_ICA_B0018=DWT_noisy_smooth(ICA_max_B18)


from scipy.stats import spearmanr
def supervised_boxcox_for_correlation(X, Y, lambdas=np.linspace(-2, 2, 100)):
    """
    Box-Cox 
    """
    best_lambda = None
    best_corr = 0
    correlations = []

    for l in lambdas:
        if l == 0:
            X_trans = np.log(X)
        else:
            X_trans = (X ** l - 1) / l
        corr, _ = spearmanr(X_trans, Y)
        correlations.append(corr)
        if abs(corr) > abs(best_corr):
            best_corr = corr
            best_lambda = l

    return best_lambda, best_corr
best_lambda, best_corr=supervised_boxcox_for_correlation(smooth_ICA_B0018,smooth_capacity_B0018)
print(best_lambda)
print(best_corr)
def boxcox_transform(X, lambda_):
    if lambda_ == 0:
        return np.log(X)
    else:
        return (X ** lambda_ - 1) / lambda_
#min max
smooth_ICA_B0018_transformed = boxcox_transform(smooth_ICA_B0018, best_lambda)
ICA_B18=min_max_normalization(smooth_ICA_B0018_transformed)
ICA_B18_exp=min_max_normalization(ICA_max_B18)
#ICA_B18_exp=ICA_max_B18
#################################################################################################################################################


















#HI4: CCCT
############################################################################################################
#B05
find_switch={}
cycle=1
t_end_B05=[]
for rows in range(data_size_B0005):
    if raw_data_B0005[rows][0][0]=="charge":
        I_charge_B05=raw_data_B0005[rows][3][0][0][1][0]
        time_b05=raw_data_B0005[rows][3][0][0][-1][0]
        t_end_B05.append(time_b05[-1])
        find_switch[f"I_Cycle_{cycle}"]=pd.Series(I_charge_B05)
        find_switch[f"T_Cylcle_{cycle}"]=pd.Series(time_b05)
        cycle+= 1
df_switch_B05=pd.DataFrame(find_switch)
#df_switch_B05.to_csv("I_T_B05.csv",index=False)


switch_point_B05=[]

for i in range(1,169):
    I=df_switch_B05.iloc[:,i*2]
    sample_point=next(i for i,a in enumerate(I) if a < 1.49 and i>2)
    T=df_switch_B05.iloc[:,(i*2)+1]
    switch_point_B05.append(T.iloc[sample_point])



z_scores = zscore(switch_point_B05)
outliers = [val for val, z in zip(switch_point_B05, z_scores) if abs(z) > 3]
#print("Outliers:", outliers)
outlier_idx=next(i for i,t in enumerate(switch_point_B05) if t in outliers)
switch_point_B05[outlier_idx]=(switch_point_B05[outlier_idx-1]+switch_point_B05[outlier_idx+1])/2

smooth_sp_B0005=DWT_noisy_smooth(switch_point_B05)
SP_B05=min_max_normalization(smooth_sp_B0005)
SP_B05_exp=min_max_normalization(switch_point_B05)
SP_B05_example=switch_point_B05


#B06
find_switch={}
cycle=1
t_end_B06=[]
for rows in range(data_size_B0006):
    if raw_data_B0006[rows][0][0]=="charge":
        I_charge_B06=raw_data_B0006[rows][3][0][0][1][0]
        time_b06=raw_data_B0006[rows][3][0][0][-1][0]
        t_end_B06.append(time_b06[-1])
        find_switch[f"I_Cycle_{cycle}"]=pd.Series(I_charge_B06)
        find_switch[f"T_Cylcle_{cycle}"]=pd.Series(time_b06)
        cycle+= 1
df_switch_B06=pd.DataFrame(find_switch)
#df_switch_B06.to_csv("I_T_B06.csv",index=False)


switch_point_B06=[]
for i in range(1,169):
    I=df_switch_B06.iloc[:,i*2]
    sample_point=next(i for i,a in enumerate(I) if a < 1.49 and i>2)
    T=df_switch_B06.iloc[:,(i*2)+1]
    switch_point_B06.append(T.iloc[sample_point])



z_scores = zscore(switch_point_B06)
outliers = [val for val, z in zip(switch_point_B06, z_scores) if abs(z) > 3]
#print("Outliers:", outliers)
#No Outliers
smooth_sp_B0006=DWT_noisy_smooth(switch_point_B06)
SP_B06=min_max_normalization(smooth_sp_B0006)
SP_B06_exp=min_max_normalization(switch_point_B06)
#SP_B06_exp=switch_point_B06

#B07
find_switch={}
cycle=1
t_end_B07=[]
for rows in range(data_size_B0007):
    if raw_data_B0007[rows][0][0]=="charge":
        I_charge_B07=raw_data_B0007[rows][3][0][0][1][0]
        time_b07=raw_data_B0007[rows][3][0][0][-1][0]
        t_end_B07.append(time_b07[-1])
        find_switch[f"I_Cycle_{cycle}"]=pd.Series(I_charge_B07)
        find_switch[f"T_Cylcle_{cycle}"]=pd.Series(time_b07)
        cycle+= 1
df_switch_B07=pd.DataFrame(find_switch)
#df_switch_B07.to_csv("I_T_B07.csv",index=False)


switch_point_B07=[]

for i in range(1,169):
    I=df_switch_B07.iloc[:,i*2]
    sample_point=next(i for i,a in enumerate(I) if a < 1.48 and i>2)
    T=df_switch_B07.iloc[:,(i*2)+1]
    switch_point_B07.append(T.iloc[sample_point])



z_scores = zscore(switch_point_B07)
outliers = [val for val, z in zip(switch_point_B07, z_scores) if abs(z) > 2]
#print("Outliers:", outliers)
#outliers
outlier_idx=[i for i,t in enumerate(switch_point_B07) if t in outliers]
#print(outlier_idx)
switch_point_B07[outlier_idx[0]]=(switch_point_B07[outlier_idx[0]-1]+switch_point_B07[outlier_idx[0]+1])/2
switch_point_B07[outlier_idx[1]]=(switch_point_B07[outlier_idx[1]-1]+switch_point_B07[outlier_idx[1]+1])/2
switch_point_B07[outlier_idx[2]]=(switch_point_B07[outlier_idx[2]-1]+switch_point_B07[outlier_idx[2]+1])/2
switch_point_B07[outlier_idx[3]]=(switch_point_B07[outlier_idx[3]-1]+switch_point_B07[outlier_idx[3]+1])/2
switch_point_B07[outlier_idx[4]]=(switch_point_B07[outlier_idx[4]-1]+switch_point_B07[outlier_idx[4]+1])/2
switch_point_B07[outlier_idx[5]]=(switch_point_B07[outlier_idx[5]-1]+switch_point_B07[outlier_idx[5]+1])/2
switch_point_B07[outlier_idx[6]]=(switch_point_B07[outlier_idx[6]-1]+switch_point_B07[outlier_idx[6]+1])/2
switch_point_B07[outlier_idx[7]]=(switch_point_B07[outlier_idx[7]-1]+switch_point_B07[outlier_idx[7]+1])/2
smooth_sp_B0007=DWT_noisy_smooth(switch_point_B07)
SP_B07=min_max_normalization(smooth_sp_B0007)
SP_B07_exp=min_max_normalization(switch_point_B07)
#SP_B07_exp=switch_point_B07


#B18
find_switch={}
cycle=1
t_end_B18=[]
for rows in range(data_size_B0018):
    if raw_data_B0018[rows][0][0]=="charge":
        I_charge_B18=raw_data_B0018[rows][3][0][0][1][0]
        time_b18=raw_data_B0018[rows][3][0][0][-1][0]
        t_end_B18.append(time_b18[-1])
        find_switch[f"I_Cycle_{cycle}"]=pd.Series(I_charge_B18)
        find_switch[f"T_Cylcle_{cycle}"]=pd.Series(time_b18)
        cycle+= 1
df_switch_B18=pd.DataFrame(find_switch)
#df_switch_B18.to_csv("I_T_B18.csv",index=False)


switch_point_B18=[]

for i in range(0,cycle_num_B18):#132
    I=df_switch_B18.iloc[:,i*2]
    sample_point=next(i for i,a in enumerate(I) if a < 1.5 and i>2)
    T=df_switch_B18.iloc[:,(i*2)+1]
    switch_point_B18.append(T.iloc[sample_point])



z_scores = zscore(switch_point_B18)
outliers = [val for val, z in zip(switch_point_B18, z_scores) if abs(z) > 3]
#print("Outliers:", outliers)
#two outliers
outlier_idx=[i for i,t in enumerate(switch_point_B18) if t in outliers]
#print(outlier_idx)
switch_point_B18[outlier_idx[0]]=(switch_point_B18[outlier_idx[0]-1]+switch_point_B18[outlier_idx[0]+1])/2
switch_point_B18[outlier_idx[1]]=(switch_point_B18[outlier_idx[1]-1]+switch_point_B18[outlier_idx[1]+1])/2
smooth_sp_B0018=DWT_noisy_smooth(switch_point_B18)
SP_B18=min_max_normalization(smooth_sp_B0018)
SP_B18_exp=min_max_normalization(switch_point_B18)
#SP_B18_exp=switch_point_B18
######################################################################################################################################################################################################################








#Data for experiment_Type 1
###########################################################################################################################
df_B05_exp = pd.DataFrame({
    'Capacity_exp': np.squeeze(C_B05_exp),
    'DVD_ETI_exp': DVD_B05_exp,
    'ICA_exp': ICA_B05_exp,
    'CCCT_exp': SP_B05_exp,
        
})

df_B06_exp = pd.DataFrame({
    'Capacity_exp': np.squeeze(C_B06_exp),
    'DVD_ETI_exp': DVD_B06_exp,
    'ICA_exp': ICA_B06_exp,
    'CCCT_exp': SP_B06_exp,
    #'CVCT': CVCT_B06,
    #'DVA': DVA_B06

})

df_B07_exp = pd.DataFrame({
    'Capacity': np.squeeze(C_B07_exp),
    'DVD_ETI': DVD_B07_exp,
    'ICA': ICA_B07_exp,
    'CCCT': SP_B07_exp,
    #'CVCT': CVCT_B07,
    #'DVA': DVA_B07
})

df_B18_exp = pd.DataFrame({
    'Capacity': np.squeeze(C_B18_exp),
    'DVD_ETI': DVD_B18_exp,
    'ICA': ICA_B18_exp,
    'CCCT': SP_B18_exp,
    #'CVCT': CVCT_B18,
    #'DVA': DVA_B18
})

#df_B05_exp.to_csv("HI_B05_exp.csv", index=False)
#df_B06_exp.to_csv("HI_B06_exp.csv", index=False)
#df_B07_exp.to_csv("HI_B07_exp.csv", index=False)
#df_B18_exp.to_csv("HI_B18_exp.csv", index=False)
########################################################################################################################################################












































