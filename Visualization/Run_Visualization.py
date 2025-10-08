import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Times New Roman'
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
#color

#2878b5   dark blue
#9ac9db   light blue
#f8ac8c    light red
#c82423    dark red
#ff8884     vine red

#This file not very important, its just for me draw Figure









"""
#Fig1: CO2 emission of German from 1977 to 2022 ->https://www.worldometers.info/co2-emissions/germany-co2-emissions/
years=[1977,1978,1979,1980,1981,1982,1983,1984,1985,1986,1987,1988,
      1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,
      2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,
      2013,2014,2015,2016,2017,2018,2019,2020,2021,2022]
population=[78.18,78.11,78.14,78.31,78.44,78.39,78.22,77.99,77.86,77.92,78.06,78.38,79.00,79.70,80.30,
            80.93,81.48,81.77,82.01,82.26,82.38,82.01,81.67,81.79,81.93,82.06,82.11,82.09,82.04,81.70,81.29,
            81.11,80.93,80.83,80.85,80.91,81.04,81.37,82.07,82.76,83.10,83.36,83.55,83.62,83.69,83.08]#Mio
emissions=[1095,1133,1182,1128,1093,1047,1064,1078,1080,1074,1066,1059,1045,1008,986,933,925,910,
          906,935,903,902,868,866,882,867,870,855,835,848,818,823,763,809,783,795,813,773,779,783,769,
          744,693,637,679,673]

sns.set_context("paper")
sns.set_style("whitegrid")
sns.set_palette("muted")

plt.rcParams["font.family"]="Times New Roman"


fig, ax1 = plt.subplots(figsize=(6, 3.5)) 


color_pop = "#2878b5"
ax1.plot(years, population, color=color_pop, label="Population", linewidth=1.5)
ax1.set_xlabel("Year", fontsize=9)
ax1.set_ylabel("Population (Mio)", color=color_pop, fontsize=9)
ax1.tick_params(axis="y", labelcolor=color_pop)


ax2 = ax1.twinx()
color_co2 = "#c82423"
ax2.plot(years, emissions, color=color_co2, label="CO₂ Emissions", linewidth=1.5, linestyle='--')
ax2.set_ylabel("CO₂ Emissions (tons)", color=color_co2, fontsize=9)
ax2.tick_params(axis="y", labelcolor=color_co2)


# ax1.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)


plt.title("Germany Population and CO₂ Emissions (1977–2022)", fontsize=11)


lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc="lower right", fontsize=7)


sns.despine(fig)
plt.tight_layout()  
#plt.savefig("germany_population_co2.pdf", format="pdf", bbox_inches="tight")
#plt.show()

"""




"""

# fig 2: 
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_context("paper")
sns.set_style("whitegrid")
plt.rcParams["font.family"] = "Times New Roman"




label1=["Hard coal","Nuclear Energy","Lignite","Renewables","Others","Oil","Natural gas"]
sizes1 = [14.1, 11.7, 22.5,33.3,4.2,0.9,13.2]
label2=["Wind Power","Municipal waste","Photovoltaics","Biomass","Hydropower"]
sizes2 = [16.3, 0.9, 6.1,6.9,3.1]

#########################################################
fig, ax1 = plt.subplots(figsize=(3.5, 3.5))  

wedges1, texts1, autotexts1 = ax1.pie(
    sizes1,
    labels=label1,
    colors=["#00749f","#95c3da","#004376","#76b728","#ffe1aa","#edab66","#00afda"],
    autopct="%1.0f%%",
    startangle=90,
    pctdistance=1.1,
    labeldistance=1.25,  
    radius=0.5
)
centre_circle1 = plt.Circle((0, 0), 0.30, fc='white')
ax1.add_artist(centre_circle1)



ax1.axis('equal')


for autotext in autotexts1:
    autotext.set_fontsize(9)

sns.despine(fig)
plt.tight_layout()
plt.show()

########################################################################
fig, ax2 = plt.subplots(figsize=(3.5, 3.5))  

wedges2, texts2, autotexts2 = ax2.pie(
    sizes2,
    labels=label2,
    colors=["#b5e0f0","#004375","#fab900","#d0e2ae","#00afda"],
    autopct="%1.0f%%",
    startangle=90,
    pctdistance=1.1,  
    labeldistance=1.25,
    radius=0.5
)
centre_circle2 = plt.Circle((0, 0), 0.30, fc='white')
ax2.add_artist(centre_circle2)



ax2.axis('equal')


for autotext in autotexts2:
    autotext.set_fontsize(9)

sns.despine(fig)
plt.tight_layout()
plt.show()
"""











#figure_10
df_discharge = pd.read_csv('Raw_Datas/All discharge/all discharge.csv')
V_discharge=df_discharge.iloc[0:178,0].tolist()
A_discharge=my_list = [-2] * len(V_discharge)
T_discharge=df_discharge.iloc[0:178,5].tolist()

#B0005
data_path_B0005=os.path.join("Raw_Datas","B0005")
load_mat_B0005=loadmat(data_path_B0005)
raw_data_B0005=load_mat_B0005["B0005"][0][0][0][0]
data_size_B0005=raw_data_B0005.shape[0]

V_charge=raw_data_B0005[0][3][0][0][0][0]
A_charge=raw_data_B0005[0][3][0][0][1][0]
T_charge=raw_data_B0005[0][3][0][0][-1][0]


T_discharge_=T_discharge+T_charge[-1]
T = np.concatenate((T_charge, T_discharge_))#789+178=967
V = np.concatenate((V_charge, V_discharge))#967
A = np.concatenate((A_charge,A_discharge))#967

#plt.rcParams['font.family'] = 'Times New Roman'
#plt.rcParams['axes.unicode_minus'] = False
#color1="#2878b5"
#color2="#c82423"
#fig,ax1=plt.subplots(figsize=(5,3),dpi=300)
#ax1.plot(T,V,color=color1,label="Voltage",linestyle="-",linewidth="1.5")
#ax1.set_xlabel("Time/s",fontsize=9)
#ax1.set_ylabel("Voltage",color=color1,fontsize=9)
#ax1.tick_params(axis="y",labelcolor=color1,labelsize=8)

#ax2= ax1.twinx()
#ax2.plot(T,A,color=color2,linestyle="--",label="Ampel",linewidth=1.5)
#ax2.set_ylabel("Ampel",color=color2,fontsize=9)
#ax2.tick_params(axis="y",labelcolor=color2,labelsize=8)

#plt.title("Charge and discharge process of first cycle(#B05)",fontsize=9,pad=5)
#plt.tight_layout()
#plt.show()

"""
#====================
#for Presentation
#====================
plt.rcParams['font.family'] = 'Segoe UI'
plt.rcParams['axes.unicode_minus'] = False
color1="#00549F"
color2="#DA1F3D"
fig,ax1=plt.subplots(figsize=(5,3),dpi=600)
ax1.plot(T,V,color=color1,label="Voltage (V)",linestyle="-",linewidth="1.5")
ax1.set_xlabel("Time (s)",fontsize=9)
ax1.set_ylabel("Voltage (V)",color=color1,fontsize=9)
ax1.tick_params(axis="y",labelcolor=color1,labelsize=8)

ax2= ax1.twinx()
ax2.plot(T,A,color=color2,linestyle="--",label="Current",linewidth=1.5)
ax2.set_ylabel("Current (A)",color=color2,fontsize=9)
ax2.tick_params(axis="y",labelcolor=color2,labelsize=8)


plt.tight_layout()
plt.savefig("VA_curve.png", dpi=600, bbox_inches="tight")
#plt.show()
"""




"""
#figure_11
from mpl_toolkits.mplot3d import Axes3D
#input
df_B05=pd.read_csv(f"Raw_Datas\capacity\B0005.csv")
df_B06=pd.read_csv(f"Raw_Datas\capacity\B0006.csv")
df_B07=pd.read_csv(f"Raw_Datas\capacity\B0007.csv")
df_B18=pd.read_csv(f"Raw_Datas\capacity\B0018.csv")


fig=plt.figure(figsize=(10,8))
ax=fig.add_subplot(1,1,1, projection='3d')
for df,label,color,cell in zip(
    [df_B05,df_B06,df_B07,df_B18],
    ["B05","B06","B07","B18"],
    ["#2878b5", "#c82423", "#3c9d5d", "#8d5fb3"],
    [1,2,3,4]
):
    ax.plot(df["cycle"],cell,df["capacity"],label=label,color=color,linewidth=2)

ax.set_xlabel("Cycle Number", fontsize=12, labelpad=10,fontname="Times New Roman")
ax.set_ylabel("Battery Index", fontsize=12, labelpad=10,fontname="Times New Roman")
ax.set_zlabel("Capacity (Ah)", fontsize=12, labelpad=10,fontname="Times New Roman")
ax.set_yticks([1, 2, 3, 4])
ax.set_yticklabels(["B05", "B06", "B07", "B18"], fontsize=10,fontname="Times New Roman")

for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontname("Times New Roman")


for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontname("Times New Roman")


for tick in ax.zaxis.get_major_ticks():
    tick.label1.set_fontname("Times New Roman")
ax.tick_params(axis='both', labelsize=10)
ax.view_init(elev=30, azim=-60) 

plt.rcParams["font.family"] = "Times New Roman"
plt.tight_layout()
plt.show()
"""

"""
#==================
#For presentation
#==================
from mpl_toolkits.mplot3d import Axes3D
#input
df_B05=pd.read_csv(f"Raw_Datas\capacity\B0005.csv")
df_B06=pd.read_csv(f"Raw_Datas\capacity\B0006.csv")
df_B07=pd.read_csv(f"Raw_Datas\capacity\B0007.csv")
df_B18=pd.read_csv(f"Raw_Datas\capacity\B0018.csv")


fig=plt.figure(figsize=(10,8))
ax=fig.add_subplot(1,1,1, projection='3d')
for df,label,color,cell in zip(
    [df_B05,df_B06,df_B07,df_B18],
    ["B05","B06","B07","B18"],
    ["#00549F","#DA1F3D",	"#7AB51D","#8EBAE5"],
    [1,2,3,4]
):
    ax.plot(df["cycle"],cell,df["capacity"],label=label,color=color,linewidth=2)

ax.set_xlabel("Cycle Number", fontsize=16, labelpad=10,fontname="Segoe UI")
ax.set_ylabel("Battery Index", fontsize=16, labelpad=10,fontname="Segoe UI")
ax.set_zlabel("Capacity (Ah)", fontsize=16, labelpad=10,fontname="Segoe UI")
ax.set_yticks([1, 2, 3, 4])
ax.set_yticklabels(["B05", "B06", "B07", "B18"], fontsize=10,fontname="Segoe UI")

for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontname("Segoe UI")


for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontname("Segoe UI")


for tick in ax.zaxis.get_major_ticks():
    tick.label1.set_fontname("Segoe UI")
ax.tick_params(axis='both', labelsize=14)
ax.view_init(elev=30, azim=-60) 

plt.rcParams["font.family"] = "Segoe UI"
plt.tight_layout()
plt.savefig("C.png", dpi=600, bbox_inches='tight')
"""





"""
#figure 12

dict={}
df = pd.read_csv('Raw_Datas/All discharge/all discharge.csv')
color=["#2878b5", "#c82423", "#3c9d5d", "#8d5fb3"]
markers=["o","s","^","D"]
linestyles=["-","--","-.",":"]
cycles=[1,40,80,120]
for cycles_nums in cycles:
    df_filter = df[(df["id_cycle"] == cycles_nums) & (df["Battery"]=="B0005")]
    V=df_filter.iloc[:,0].to_list()
    T=df_filter.iloc[:,5].to_list()
    dict[cycles_nums]={"V":V,"T":T}
for idx, (keys,values) in enumerate(dict.items()):
    plt.scatter(values["T"],values["V"], s=2,alpha=0.8,
                color=color[idx],
                marker=markers[idx],
                linestyle=linestyles[idx],
                linewidths=1.5,
                label=f"Cycle {keys}"
                )

plt.axvline(x=19, color='black', linestyle='--', linewidth=0.7)
plt.axvline(x=2300, color='black', linestyle='--', linewidth=0.7)
plt.xlabel("Time (s)")
plt.ylabel("Voltage (v)")
plt.legend(frameon=False, fontsize=10)
plt.tight_layout()
#plt.savefig("Figure_12.png", dpi=600, bbox_inches='tight')
"""

"""
#===============
#For Presentation
#==============


dict={}
df = pd.read_csv('Raw_Datas/All discharge/all discharge.csv')
color=["#00549F","#DA1F3D","#7AB51D","#8EBAE5"]
markers=["o","s","^","D"]
linestyles=["-","--","-.",":"]
cycles=[1,40,80,120]
for cycles_nums in cycles:
    df_filter = df[(df["id_cycle"] == cycles_nums) & (df["Battery"]=="B0005")]
    V=df_filter.iloc[:,0].to_list()
    T=df_filter.iloc[:,5].to_list()
    dict[cycles_nums]={"V":V,"T":T}
for idx, (keys,values) in enumerate(dict.items()):
    plt.scatter(values["T"],values["V"], s=2,alpha=0.8,
                color=color[idx],
                marker=markers[idx],
                linestyle=linestyles[idx],
                linewidths=1.5,
                label=f"Cycle {keys}"
                )

plt.axvline(x=19, color='black', linestyle='--', linewidth=0.7)
plt.axvline(x=2300, color='black', linestyle='--', linewidth=0.7)
plt.xlabel("Time (s)", fontname="Segoe UI",fontsize=16)
plt.ylabel("Voltage (V)", fontname="Segoe UI",fontsize=16)
plt.tick_params(axis='both', labelsize=14)
plt.legend(frameon=False, fontsize=14, prop={"family": "Segoe UI"})
plt.tight_layout()
plt.savefig("DVD_ETI.png", dpi=600, bbox_inches='tight')
plt.rcParams["font.family"] = "Segoe UI"
"""










"""
#Figure_13: ICA 1 40 80 120 160
import pywt
import matplotlib.ticker as ticker
def DWT_noisy_smooth(raw_HI, wavelet="db4", level=5):
    coeffs = pywt.wavedec(raw_HI, wavelet, level=level)
    coeffs[1:] = [np.zeros_like(detail) for detail in coeffs[1:]]
    trend_only = pywt.waverec(coeffs, wavelet) 
    return trend_only[:len(raw_HI)]

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
        origin_ICA=(np.diff(time)/ dU)*np.array(current[1:]) / 3600
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


y1 = combined_df_B05.iloc[:, 0]  # 第一列：ICA_Cycle_1
x1 = combined_df_B05.iloc[:, 1]  # 第二列：Voltage_Cycle_1

y40= combined_df_B05.iloc[:, 78]
x40= combined_df_B05.iloc[:, 79]

y80= combined_df_B05.iloc[:, 158]
x80= combined_df_B05.iloc[:, 159]

y120= combined_df_B05.iloc[:, 238]
x120= combined_df_B05.iloc[:, 239]

y160 = combined_df_B05.iloc[:, 318]
x160 = combined_df_B05.iloc[:, 319]

cycles = {
    'Cycle 1': (x1, y1),
    'Cycle 40': (x40, y40),
    'Cycle 80': (x80, y80),
    'Cycle 120': (x120, y120),
    'Cycle 160': (x160, y160),
}

color = ["#2878b5", "#c82423", "#3c9d5d", "#8d5fb3", "#f39c12"]
markers = ["o", "s", "^", "D", "v"]

for idx,(keys,values) in enumerate(cycles.items()):
    plt.plot(values[0],values[1],alpha=0.8,
             color=color[idx],
             marker=markers[idx],
             markersize=2,
             label=keys,
             linewidth=0.5
             )

plt.ylabel("IC [Ah/V]")
plt.xlabel("Voltage [V]")
plt.legend(frameon=False, fontsize=10)
plt.tight_layout()
#plt.show()
#plt.savefig("Figure_13.png", dpi=600, bbox_inches='tight')
"""





"""
#==================
#For Presentation
#==================
import pywt
import matplotlib.ticker as ticker
def DWT_noisy_smooth(raw_HI, wavelet="db4", level=5):
    coeffs = pywt.wavedec(raw_HI, wavelet, level=level)
    coeffs[1:] = [np.zeros_like(detail) for detail in coeffs[1:]]
    trend_only = pywt.waverec(coeffs, wavelet) 
    return trend_only[:len(raw_HI)]

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
        origin_ICA=(np.diff(time)/ dU)*np.array(current[1:]) / 3600
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



y1 = combined_df_B05.iloc[:, 0]  # 第一列：ICA_Cycle_1
x1 = combined_df_B05.iloc[:, 1]  # 第二列：Voltage_Cycle_1

y40= combined_df_B05.iloc[:, 78]
x40= combined_df_B05.iloc[:, 79]

y80= combined_df_B05.iloc[:, 158]
x80= combined_df_B05.iloc[:, 159]

y120= combined_df_B05.iloc[:, 238]
x120= combined_df_B05.iloc[:, 239]

y160 = combined_df_B05.iloc[:, 318]
x160 = combined_df_B05.iloc[:, 319]

cycles = {
    'Cycle 1': (x1, y1),
    'Cycle 40': (x40, y40),
    'Cycle 80': (x80, y80),
    'Cycle 120': (x120, y120),
    'Cycle 160': (x160, y160),
}

color = ["#00549F","#DA1F3D","#7AB51D","#8EBAE5", "#F29400"]
markers = ["o", "s", "^", "D", "v"]

for idx,(keys,values) in enumerate(cycles.items()):
    plt.plot(values[0],values[1],alpha=0.8,
             color=color[idx],
             marker=markers[idx],
             markersize=2,
             label=keys,
             linewidth=0.5
             )

plt.ylabel("IC (Ah/V)", fontname="Segoe UI",fontsize=16)
plt.xlabel("Voltage (V)", fontname="Segoe UI",fontsize=16)
plt.tick_params(axis='both', labelsize=14)
plt.legend(frameon=False, fontsize=10, prop={"family": "Segoe UI"})
plt.tight_layout()
plt.savefig("ICA.png", dpi=600, bbox_inches='tight')

plt.rcParams["font.family"] = "Segoe UI"
"""








"""
#Figure_14: CCCT 1 40 60 80 120 160
#B05
I_charge_cycle= {}
time_charge_cycle={}
cycle=1
for rows in range(data_size_B0005):
    if raw_data_B0005[rows][0][0]=="charge":
        I_charge_B05=raw_data_B0005[rows][3][0][0][1][0]
        time_b05=raw_data_B0005[rows][3][0][0][-1][0]
        I_charge_cycle[cycle]=I_charge_B05
        time_charge_cycle[cycle]=time_b05
        cycle +=1

selected_cycles = [1, 40, 80, 120, 160]

colors = ["#2878b5", "#c82423", "#3c9d5d", "#8d5fb3", "#f39c12"]
markers = ["o", "s", "^", "D", "v"]

plt.figure(figsize=(8, 6))
for i, cycle in enumerate(selected_cycles):
    time = time_charge_cycle[cycle]
    current = I_charge_cycle[cycle]
    plt.plot(time, current, label=f'Cycle {cycle}', color=colors[i],marker=markers[i],
             markersize=0.5,
             linewidth=0.5,)
plt.xlabel("Time [s]")
plt.ylabel("Charging Current [A]")
plt.legend(frameon=False, fontsize=10)
plt.tight_layout()
#plt.show()
plt.savefig("Figure_14.png", dpi=600, bbox_inches='tight')
"""
"""
#=========================
# for Presentation
#============================
I_charge_cycle= {}
time_charge_cycle={}
cycle=1
for rows in range(data_size_B0005):
    if raw_data_B0005[rows][0][0]=="charge":
        I_charge_B05=raw_data_B0005[rows][3][0][0][1][0]
        time_b05=raw_data_B0005[rows][3][0][0][-1][0]
        I_charge_cycle[cycle]=I_charge_B05
        time_charge_cycle[cycle]=time_b05
        cycle +=1

selected_cycles = [1, 40, 80, 120, 160]


colors = ["#00549F","#DA1F3D","#7AB51D","#8EBAE5", "#F29400"]
markers = ["o", "s", "^", "D", "v"]


plt.figure(figsize=(8, 6))
for i, cycle in enumerate(selected_cycles):
    time = time_charge_cycle[cycle]
    current = I_charge_cycle[cycle]
    plt.plot(time, current, label=f'Cycle {cycle}', color=colors[i],marker=markers[i],
             markersize=0.5,
             linewidth=0.5,)
plt.xlabel("Time (s)", fontname="Segoe UI", fontsize=16)
plt.ylabel("Charging Current (A)", fontname="Segoe UI",fontsize=16)
plt.legend(frameon=False, fontsize=14, prop={"family": "Segoe UI"})
plt.tick_params(axis='both', labelsize=14)
plt.tight_layout()
plt.savefig("CCCT.png", dpi=600, bbox_inches='tight')

plt.rcParams["font.family"] = "Segoe UI"
"""

"""
#Figure 15:DWT
import pywt
df_B0005 = pd.read_csv('Raw_Datas/capacity/B0005.csv')
raw_capacity_B0005 = df_B0005['capacity'].values
N=len(raw_capacity_B0005)
color = "#2878b5"
color2="#c82423"
# 小波分解
wavelet = "db4"
level = 4
coeffs = pywt.wavedec(raw_capacity_B0005, wavelet, level=level)


plt.figure(figsize=(10,6))
plt.plot(range(N),raw_capacity_B0005,color=color,label="Raw Capacity (#B05)",linewidth=2)
plt.legend(fontsize=18)
plt.ylabel("Capacity (Ah)", fontsize=18)
plt.xlabel("Cycle Number", fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.grid(False)
plt.savefig("Figure_a.png", dpi=600, bbox_inches='tight')
#################################################################################################################


# 可视化每一层系数 figure_b
plt.figure(figsize=(10, 6))

for i, coef in enumerate(coeffs[::-1]):
    #x_old=np.linspace(0,N,len(coef))
    #x_new=np.linspace(0,N,N)
    #coef=np.interp(x_new,x_old,coef)
    plt.subplot(level+1, 1, i+1)
    plt.plot(coef, linewidth=1.5,color=color)
    plt.grid(False)


plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
#######################################################################################################
def DWT_noisy_smooth(raw_HI, wavelet="db4", level=5):
    coeffs = pywt.wavedec(raw_HI, wavelet, level=level)
    coeffs[1:] = [np.zeros_like(detail) for detail in coeffs[1:]]
    trend_only = pywt.waverec(coeffs, wavelet) 
    return trend_only[:len(raw_HI)]
smooth_capacity_B0005=DWT_noisy_smooth(raw_capacity_B0005)

plt.figure(figsize=(10,6))
plt.plot(range(N),raw_capacity_B0005,color=color,label="Raw Capacity (#B05)",linewidth=1,linestyle="--")
plt.plot(smooth_capacity_B0005, label="Reconstructed Capacity", linewidth=2,color=color2)

plt.legend(fontsize=18)
plt.xlabel("Cycle Number", fontsize=18)
plt.ylabel("Capacity (Ah)", fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.tight_layout()
plt.grid(False)
plt.savefig("Figure_c.png", dpi=600, bbox_inches='tight')
"""

r"""
#==================
#For presentation
#=========================
import pywt
df_B0005 = pd.read_csv('Raw_Datas/capacity/B0005.csv')
raw_capacity_B0005 = df_B0005['capacity'].values
N=len(raw_capacity_B0005)
color = "#00549F"
color2="#DA1F3D"

wavelet = "db4"
level = 4
coeffs = pywt.wavedec(raw_capacity_B0005, wavelet, level=level)


plt.figure(figsize=(10,6))
plt.plot(range(N),raw_capacity_B0005,color=color,label="Raw Capacity (#B05)",linewidth=2)
plt.legend()
plt.ylabel("Capacity (Ah)")
plt.xlabel("Cycle Number")
plt.tight_layout()
plt.grid(False)
plt.savefig("Figure_a.png", dpi=600, bbox_inches='tight')

#################################################################################################################



plt.figure(figsize=(10, 6))

for i, coef in enumerate(coeffs[::-1]):
    #x_old=np.linspace(0,N,len(coef))
    #x_new=np.linspace(0,N,N)
    #coef=np.interp(x_new,x_old,coef)
    plt.subplot(level+1, 1, i+1)
    plt.plot(coef, linewidth=1.5,color=color)
    plt.grid(False)


plt.tight_layout(rect=[0, 0, 1, 0.96])
#plt.show()
#######################################################################################################
def DWT_noisy_smooth(raw_HI, wavelet="db4", level=5):
    coeffs = pywt.wavedec(raw_HI, wavelet, level=level)
    coeffs[1:] = [np.zeros_like(detail) for detail in coeffs[1:]]
    trend_only = pywt.waverec(coeffs, wavelet) 
    return trend_only[:len(raw_HI)]
smooth_capacity_B0005=DWT_noisy_smooth(raw_capacity_B0005)

#plt.figure(figsize=(10,6))
#plt.plot(range(N),raw_capacity_B0005,color=color,label="Raw Capacity (#B05)",linewidth=1,linestyle="--")
#plt.plot(smooth_capacity_B0005, label="Reconstructed Capacity", linewidth=2,color=color2)
#plt.legend(fontsize=12)
#plt.ylabel("Capacity (Ah)")
#plt.xlabel("Cycle Index")
#plt.tight_layout()
#plt.grid(False)
#plt.savefig("c_compare.png", dpi=600, bbox_inches='tight')
"""


"""
#Figure 19: Epsilon
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "Times New Roman",  
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300
})


epochs = np.arange(0, 300)
label=np.arange(50,350)

epsilon = np.minimum(0.5, 1 - 0.9 * np.exp(-0.01 * epochs))


fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(
    label, epsilon,
    label=r'$\epsilon = \min(0.5,\ 1 - 0.9 \cdot e^{-0.01 \cdot \mathrm{epoch}})$',
    color='#2878b5',
    linewidth=2
)


ax.set_xlabel('Epoch')
ax.set_ylabel(r'$\epsilon$')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(frameon=False)
ax.legend(frameon=False, prop={'family': 'Times New Roman', 'size': 12})


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

plt.savefig("Figure_19.png", dpi=300, bbox_inches='tight')


#plt.show()
"""




r"""
#Figure 20:Transfer learning
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.pipeline import make_pipeline


plt.rcParams["font.family"] = "Times New Roman"


path_B05 = r"C:\Users\zheng\Desktop\RUL_Battery\Data for model\HI_B05.csv"
df_raw = pd.read_csv(path_B05)
C_05 = df_raw.iloc[:, 0].values


degree = 3
alpha = 0.1
epsilon = 0.10

def generate_pseudo_curve_only(data):
    x = np.arange(len(data)).reshape(-1, 1)
    y = np.array(data)

    model = make_pipeline(
        PolynomialFeatures(degree=degree),
        Ridge(alpha=alpha)
    )
    model.fit(x, y)

    ridge_coef = model.named_steps['ridge'].coef_
    ridge_intercept = model.named_steps['ridge'].intercept_

    np.random.seed(42)
    perturbed_coef = ridge_coef * (1 + epsilon * np.random.randn(len(ridge_coef)))
    perturbed_intercept = ridge_intercept * (1 + epsilon * np.random.randn())

    x_poly = PolynomialFeatures(degree=degree).fit_transform(x)
    pseudo_curve = np.dot(x_poly, perturbed_coef) + perturbed_intercept

    if y[40] > y[50]:
        for i in range(1, len(pseudo_curve)):
            if pseudo_curve[i] >= pseudo_curve[i - 1]:
                pseudo_curve[i] = pseudo_curve[i - 1] * 0.998

    scaler = MinMaxScaler()
    pseudo_curve = scaler.fit_transform(pseudo_curve.reshape(-1, 1)).flatten()

    return pseudo_curve

pseudo_curve_C = generate_pseudo_curve_only(C_05)


plt.figure(figsize=(8, 4))
plt.plot(C_05, color="#2878b5", linewidth=2)
plt.xlabel('Cycle Index')
plt.ylabel('Health Index')
plt.title('Original C_05')
plt.tight_layout()
#plt.savefig("original_C05.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(pseudo_curve_C, color="#c82423", linewidth=2, label="Pseudo C_05")
plt.plot(C_05, color="#2878b5", linestyle='--', linewidth=1.5, alpha=0.5, label="Real C_05")
plt.xlabel('Cycle Index')
plt.ylabel('Health Index')
plt.title('Pseudo C_05 vs Real C_05')
plt.legend()
plt.tight_layout()
plt.savefig("pseudo_vs_real_C05.png", dpi=300)
plt.show()
"""




r"""
#Figure_21: compare epsilon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.pipeline import make_pipeline


plt.rcParams["font.family"] = "Times New Roman"


path_B05 = r"C:\Users\zheng\Desktop\RUL_Battery\Data for model\HI_B05.csv"
df_raw = pd.read_csv(path_B05)
C_05 = df_raw.iloc[:, 0].values


degree = 3
alpha = 0.1
epsilons = [0.05, 0.10, 0.15]  
colors = ["#2878b5", "#3c9d5d","#c82423", "#8d5fb3"]  

def generate_pseudo_curve(data, epsilon):
    x = np.arange(len(data)).reshape(-1, 1)
    y = np.array(data)

    model = make_pipeline(
        PolynomialFeatures(degree=degree),
        Ridge(alpha=alpha)
    )
    model.fit(x, y)

    ridge_coef = model.named_steps['ridge'].coef_
    ridge_intercept = model.named_steps['ridge'].intercept_

    np.random.seed(42)
    perturbed_coef = ridge_coef * (1 + epsilon * np.random.randn(len(ridge_coef)))
    perturbed_intercept = ridge_intercept * (1 + epsilon * np.random.randn())

    x_poly = PolynomialFeatures(degree=degree).fit_transform(x)
    pseudo_curve = np.dot(x_poly, perturbed_coef) + perturbed_intercept

    
    if y[40] > y[50]:
        for i in range(1, len(pseudo_curve)):
            if pseudo_curve[i] >= pseudo_curve[i - 1]:
                pseudo_curve[i] = pseudo_curve[i - 1] * 0.998

    # MinMax
    scaler = MinMaxScaler()
    pseudo_curve = scaler.fit_transform(pseudo_curve.reshape(-1, 1)).flatten()

    return pseudo_curve


pseudo_curves = [generate_pseudo_curve(C_05, eps) for eps in epsilons]


plt.figure(figsize=(10, 5))


plt.plot(C_05, color=colors[0], linewidth=2, label="Real B05")


for i, eps in enumerate(epsilons):
    plt.plot(pseudo_curves[i], color=colors[i+1], linewidth=2, alpha=0.7,
             label=f"Pseudo B05 ($\\epsilon$={eps})")


plt.xlabel("Cycle Index",fontsize=18)
plt.ylabel("Normalized Capacity",fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.grid(False)
plt.tight_layout()
plt.savefig("pseudo_b05_comparison.png", dpi=600)  # 可选保存
plt.show()
"""




"""
#Figure_22: Compare all results
def min_max_inverse(normalized_HI, min_val, max_val):
    arr = np.asarray(normalized_HI, dtype=float)
    return arr * (max_val - min_val) + min_val

# ==== 配置 ====
CSV_PATH = "B05_results.csv"
COLS = ["Label_B05", "Total_B05_CNN", "Total_B05_Transformer", "Total_B05_TwinLstm"]
COLORS = {
    "label": "#2878b5",
    "cnn":   "#3c9d5d",
    "trf":   "#c82423",
    "twin":  "#8d5fb3",
}


df = pd.read_csv(CSV_PATH)


y_label_n = df["Label_B05"].to_numpy()
y_cnn_n   = df["Total_B05_CNN"].to_numpy()
y_trf_n   = df["Total_B05_Transformer"].to_numpy()
y_twin_n  = df["Total_B05_TwinLstm"].to_numpy()

# ====  min/max ====
min_val = 1.3250793286429356
max_val = 1.8564874208181574


y_label = min_max_inverse(y_label_n, min_val, max_val)
y_cnn   = min_max_inverse(y_cnn_n,   min_val, max_val)
y_trf   = min_max_inverse(y_trf_n,   min_val, max_val)
y_twin  = min_max_inverse(y_twin_n,  min_val, max_val)


n = len(df)
x = np.arange(1, n + 1, dtype=int)
mark_every = max(1, n // 20)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = 1.2

fig = plt.figure(figsize=(8.5, 5.3))
ax = plt.gca()


ax.plot(x, y_label, label="Label (B05)",
        color=COLORS["label"], linewidth=2.0, linestyle="-",
        marker="o", markersize=4, markevery=mark_every)
ax.plot(x, y_cnn, label="CNN Baseline",
        color=COLORS["cnn"], linewidth=2.0, linestyle="--",
        marker="s", markersize=4, markevery=mark_every)
ax.plot(x, y_trf, label="Transformer Baseline",
        color=COLORS["trf"], linewidth=2.0, linestyle="-.",
        marker="^", markersize=4, markevery=mark_every)
ax.plot(x, y_twin, label="Twin LSTM (Proposed)",
        color=COLORS["twin"], linewidth=2.0, linestyle=":",
        marker="D", markersize=4, markevery=mark_every)


y_all = np.concatenate([y_label, y_cnn, y_trf, y_twin])
y_min, y_max = float(np.nanmin(y_all)), float(np.nanmax(y_all))
margin = (y_max - y_min) * 0.02 if y_max > y_min else 0.01
ax.set_ylim(y_min - margin, y_max + margin)


x_start = 30 if n >= 30 else n
ax.axvline(x=x_start, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
ax.text(x_start + 0.5, y_max, "Start Point", color="gray", fontsize=12,
        rotation=0, va="bottom", ha="left")



eol_value = 2.0 * 0.7  # 1.4 Ah
ax.axhline(y=eol_value, color="black", linestyle="-", linewidth=1.0,alpha=0.6)
ax.text(x[-1], eol_value + (y_max - y_min) * 0.01, "EOL (70%)", color="black",
        fontsize=12, va="bottom", ha="right",alpha=0.8)




ax.set_xlabel("Cycle", fontsize=14)
ax.set_ylabel("Capacity (Ah)", fontsize=14)  
ax.tick_params(axis="both", which="both", direction="in", labelsize=12)
ax.minorticks_on()
ax.tick_params(which="major", length=6)
ax.tick_params(which="minor", length=3)


ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


ax.legend(fontsize=12, frameon=False, loc="best")

plt.tight_layout()
#plt.savefig("Figure_22.png", dpi=600, bbox_inches="tight")
#plt.show()



#Figure 23: AE between prediction and label
err_cnn = np.abs(y_cnn - y_label)[30:]
err_trf = np.abs(y_trf - y_label)[30:]
err_twin = np.abs(y_twin - y_label)[30:]

x_err = np.arange(30, 30 + len(err_cnn))

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = 1.2

fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.5), gridspec_kw={"height_ratios": [1.6, 1]})


ax1 = axes[0]
ax1.plot(x_err, err_cnn, label="AE - CNN Baseline",
         color=COLORS["cnn"], linewidth=2.0, linestyle="--",
         marker="s", markersize=4, markevery=5)
ax1.plot(x_err, err_trf, label="AE - Transformer Baseline",
         color=COLORS["trf"], linewidth=2.0, linestyle="-.",
         marker="^", markersize=4, markevery=5)
ax1.plot(x_err, err_twin, label="AE - Twin LSTM (Proposed)",
         color=COLORS["twin"], linewidth=2.0, linestyle=":",
         marker="D", markersize=4, markevery=5)


y_all_err = np.concatenate([err_cnn, err_trf, err_twin])
y_min, y_max = float(np.nanmin(y_all_err)), float(np.nanmax(y_all_err))
margin = (y_max - y_min) * 0.02 if y_max > y_min else 0.01
ax1.set_ylim(y_min - margin, y_max + margin)

ax1.set_ylabel("Absolute Error (Ah)", fontsize=14)
ax1.tick_params(axis="both", which="both", direction="in", labelsize=12)
ax1.minorticks_on()
ax1.tick_params(which="major", length=6)
ax1.tick_params(which="minor", length=3)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.legend(fontsize=12, frameon=False, loc="best")


ax2 = axes[1]
data = [err_cnn, err_trf, err_twin]
labels = ["CNN Baseline", "Transformer Baseline", "Twin LSTM (Proposed)"]
colors = [COLORS["cnn"], COLORS["trf"], COLORS["twin"]]

box = ax2.boxplot(
    data,
    patch_artist=True,
    labels=labels,
    widths=0.6,
    medianprops=dict(color="black", linewidth=1.5),
    whiskerprops=dict(color="black", linewidth=1.2),
    capprops=dict(color="black", linewidth=1.2),
    boxprops=dict(linewidth=1.2),
    flierprops=dict(marker="o", markerfacecolor="gray", markersize=4, alpha=0.6)
)

for patch, color in zip(box["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)

ax2.set_ylabel("Absolute Error (Ah)", fontsize=14)
ax2.tick_params(axis="both", which="both", direction="in", labelsize=12)
ax2.minorticks_on()
ax2.tick_params(which="major", length=6)
ax2.tick_params(which="minor", length=3)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)


#plt.tight_layout()
#plt.savefig("Figure_23.png", dpi=600, bbox_inches="tight")
#plt.show()
"""


r"""
#================
#For presentation
#================
def min_max_inverse(normalized_HI, min_val, max_val):
    arr = np.asarray(normalized_HI, dtype=float)
    return arr * (max_val - min_val) + min_val


CSV_PATH = "B05_results.csv"
COLS = ["Label_B05", "Total_B05_CNN", "Total_B05_Transformer", "Total_B05_TwinLstm"]
COLORS = {
    "label": "#00549F",
    "cnn":   "#DA1F3D",
    "trf":   "#7AB51D",
    "twin":  "#F29400",
}


df = pd.read_csv(CSV_PATH)


y_label_n = df["Label_B05"].to_numpy()
y_cnn_n   = df["Total_B05_CNN"].to_numpy()
y_trf_n   = df["Total_B05_Transformer"].to_numpy()
y_twin_n  = df["Total_B05_TwinLstm"].to_numpy()


min_val = 1.3250793286429356
max_val = 1.8564874208181574


y_label = min_max_inverse(y_label_n, min_val, max_val)
y_cnn   = min_max_inverse(y_cnn_n,   min_val, max_val)
y_trf   = min_max_inverse(y_trf_n,   min_val, max_val)
y_twin  = min_max_inverse(y_twin_n,  min_val, max_val)


n = len(df)
x = np.arange(1, n + 1, dtype=int)
mark_every = max(1, n // 20)


plt.rcParams["font.family"] = "Segoe UI"
plt.rcParams["axes.linewidth"] = 1.2

fig = plt.figure(figsize=(8.5, 5.3))
ax = plt.gca()


ax.plot(x, y_label, label="Label (B05)",
        color=COLORS["label"], linewidth=2.0, linestyle="-",
        marker="o", markersize=4, markevery=mark_every)
ax.plot(x, y_cnn, label="CNN Baseline",
        color=COLORS["cnn"], linewidth=2.0, linestyle="--",
        marker="s", markersize=4, markevery=mark_every)
ax.plot(x, y_trf, label="Transformer Baseline",
        color=COLORS["trf"], linewidth=2.0, linestyle="-.",
        marker="^", markersize=4, markevery=mark_every)
ax.plot(x, y_twin, label="Twin LSTM (Proposed)",
        color=COLORS["twin"], linewidth=2.0, linestyle=":",
        marker="D", markersize=4, markevery=mark_every)


y_all = np.concatenate([y_label, y_cnn, y_trf, y_twin])
y_min, y_max = float(np.nanmin(y_all)), float(np.nanmax(y_all))
margin = (y_max - y_min) * 0.02 if y_max > y_min else 0.01
ax.set_ylim(y_min - margin, y_max + margin)


x_start = 30 if n >= 30 else n
ax.axvline(x=x_start, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
ax.text(x_start + 0.5, y_max, "Start Point", color="gray", fontsize=14,
        rotation=0, va="bottom", ha="left")



eol_value = 2.0 * 0.7  # 1.4 Ah
ax.axhline(y=eol_value, color="black", linestyle="-", linewidth=1.0,alpha=0.6)
ax.text(x[-1], eol_value + (y_max - y_min) * 0.01, "EOL (70%)", color="black",
        fontsize=14, va="bottom", ha="right",alpha=0.8)




ax.set_xlabel("Cycle", fontsize=16)
ax.set_ylabel("Capacity (Ah)", fontsize=16) 
ax.tick_params(axis="both", which="both", direction="in", labelsize=16)
ax.minorticks_on()
ax.tick_params(which="major", length=6,labelsize=16)
ax.tick_params(which="minor", length=3)
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontfamily("Segoe UI")


ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


ax.legend(fontsize=14, frameon=False, loc="best")

plt.tight_layout()
plt.savefig("Results_curve.png", dpi=600, bbox_inches="tight")
#plt.show()



#Figure 23: AE between prediction and label
err_cnn = np.abs(y_cnn - y_label)[30:]
err_trf = np.abs(y_trf - y_label)[30:]
err_twin = np.abs(y_twin - y_label)[30:]

x_err = np.arange(30, 30 + len(err_cnn))

plt.rcParams["font.family"] = "Segoe UI"
plt.rcParams["axes.linewidth"] = 1.2

fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.5), gridspec_kw={"height_ratios": [1.6, 1]})


ax1 = axes[0]
ax1.plot(x_err, err_cnn, label="AE - CNN Baseline",
         color=COLORS["cnn"], linewidth=2.0, linestyle="--",
         marker="s", markersize=4, markevery=5)
ax1.plot(x_err, err_trf, label="AE - Transformer Baseline",
         color=COLORS["trf"], linewidth=2.0, linestyle="-.",
         marker="^", markersize=4, markevery=5)
ax1.plot(x_err, err_twin, label="AE - Twin LSTM (Proposed)",
         color=COLORS["twin"], linewidth=2.0, linestyle=":",
         marker="D", markersize=4, markevery=5)


y_all_err = np.concatenate([err_cnn, err_trf, err_twin])
y_min, y_max = float(np.nanmin(y_all_err)), float(np.nanmax(y_all_err))
margin = (y_max - y_min) * 0.02 if y_max > y_min else 0.01
ax1.set_ylim(y_min - margin, y_max + margin)

ax1.set_ylabel("Absolute Error (Ah)", fontsize=16)
ax1.tick_params(axis="both", which="both", direction="in", labelsize=16)
ax1.minorticks_on()
ax1.tick_params(which="major", length=6,labelsize=16)
ax1.tick_params(which="minor", length=3)
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontfamily("Segoe UI")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.legend(fontsize=14, frameon=False, loc="best")


ax2 = axes[1]
data = [err_cnn, err_trf, err_twin]
labels = ["CNN Baseline", "Transformer Baseline", "Twin LSTM (Proposed)"]
colors = [COLORS["cnn"], COLORS["trf"], COLORS["twin"]]

box = ax2.boxplot(
    data,
    patch_artist=True,
    labels=labels,
    widths=0.6,
    medianprops=dict(color="black", linewidth=1.5),
    whiskerprops=dict(color="black", linewidth=1.2),
    capprops=dict(color="black", linewidth=1.2),
    boxprops=dict(linewidth=1.2),
    flierprops=dict(marker="o", markerfacecolor="gray", markersize=4, alpha=0.6)
)

for patch, color in zip(box["boxes"], colors):
    patch.set_facecolor(color)


ax2.set_ylabel("Absolute Error (Ah)", fontsize=16)
ax2.tick_params(axis="both", which="both", direction="in", labelsize=16)
ax2.minorticks_on()
ax2.tick_params(which="major", length=6,labelsize=16)
ax2.tick_params(which="minor", length=3)
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontfamily("Segoe UI")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)


plt.tight_layout()
plt.savefig("AE.png", dpi=600, bbox_inches="tight")
"""



r"""
# Figure_24 & 25

file_path = r"C:\Users\zheng\Desktop\RUL_Battery\Experiment\different_start_points\exp_result_SP.csv"
df = pd.read_csv(file_path, header=None, names=["Windowsize", "MAE", "RMSE", "R2"])
df["Windowsize"] = df["Windowsize"].astype(str)


plt.rcParams["font.family"] = "Times New Roman"


palette_colors = {
    "10": "#2878b5",
    "30": "#3c9d5d",
    "50": "#c82423"
}

# ===================
# Figure 24: MAE
# ===================
fig, axes = plt.subplots(2, 1, figsize=(10, 6))


sns.boxplot(ax=axes[0], x="Windowsize", y="MAE", data=df, palette=palette_colors)

means = df.groupby("Windowsize")["MAE"].mean()
for i, (ws, mean_val) in enumerate(means.items()):
    axes[0].scatter(i, mean_val, color="black", s=70, marker='^', zorder=3, label="Mean" if i == 0 else "")
    axes[0].text(i, mean_val + 0.005, f"{mean_val:.3f}", ha='center', va='bottom', fontsize=10)

axes[0].set_title("Boxplot of MAE for Different Windowsize", fontsize=12)
axes[0].set_ylabel("MAE Score", fontsize=14)
axes[0].set_xlabel("")
axes[0].tick_params(axis='x', labelsize=12)
axes[0].tick_params(axis='y', labelsize=12)
axes[0].legend()


sns.violinplot(ax=axes[1], x="Windowsize", y="MAE", data=df, inner="box", palette=palette_colors)
axes[1].set_title("Violin Plot of MAE for Different Windowsize", fontsize=12)
axes[1].set_ylabel("MAE Score", fontsize=14)
axes[1].set_xlabel("Windowsize", fontsize=14)
axes[1].tick_params(axis='x', labelsize=12)
axes[1].tick_params(axis='y', labelsize=12)

plt.tight_layout()
plt.savefig("Figure_24.png", dpi=600, bbox_inches="tight")

# ===================
# Figure 25: R²
# ===================
fig, axes = plt.subplots(2, 1, figsize=(10, 6))

sns.boxplot(ax=axes[0], x="Windowsize", y="R2", data=df, palette=palette_colors)

means_r2 = df.groupby("Windowsize")["R2"].mean()
for i, (ws, mean_val) in enumerate(means_r2.items()):
    axes[0].scatter(i, mean_val, color="black", s=70, marker='^', zorder=3, label="Mean" if i == 0 else "")
    axes[0].text(i, mean_val + 0.005, f"{mean_val:.3f}", ha='center', va='bottom', fontsize=10)

axes[0].set_title("Boxplot of R² for Different Windowsize", fontsize=12)
axes[0].set_ylabel("R² Score", fontsize=14)
axes[0].set_xlabel("")
axes[0].tick_params(axis='x', labelsize=12)
axes[0].tick_params(axis='y', labelsize=12)
axes[0].legend()

sns.violinplot(ax=axes[1], x="Windowsize", y="R2", data=df, inner="box", palette=palette_colors)
axes[1].set_title("Violin Plot of R² for Different Windowsize", fontsize=12)
axes[1].set_ylabel("R² Score", fontsize=14)
axes[1].set_xlabel("Windowsize", fontsize=14)
axes[1].tick_params(axis='x', labelsize=12)
axes[1].tick_params(axis='y', labelsize=12)

plt.tight_layout()
plt.savefig("Figure_25.png", dpi=600, bbox_inches="tight")
"""

r"""
#===============
#For presentation
#==================

file_path = r"C:\Users\zheng\Desktop\RUL_Battery\Experiment\different_start_points\exp_result_SP.csv"
df = pd.read_csv(file_path, header=None, names=["Windowsize", "MAE", "RMSE", "R2"])
df["Windowsize"] = df["Windowsize"].astype(str)


plt.rcParams["font.family"] = "Segoe UI"


palette_colors = {
    "10": "#00549F",
    "30": "#7AB51D",
    "50": "#DA1F3D"
}

# ===================
# Figure 24: MAE
# ===================
fig, axes = plt.subplots(2, 1, figsize=(10, 6))


sns.boxplot(ax=axes[0], x="Windowsize", y="MAE", data=df, palette=palette_colors,saturation=1)

means = df.groupby("Windowsize")["MAE"].mean()
for i, (ws, mean_val) in enumerate(means.items()):
    axes[0].scatter(i, mean_val, color="black", s=70, marker='^', zorder=3, label="Mean" if i == 0 else "")
    axes[0].text(i, mean_val + 0.005, f"{mean_val:.3f}", ha='center', va='bottom', fontsize=16)

axes[0].set_title("Boxplot of MAE for Different Windowsize", fontsize=16)
axes[0].set_ylabel("MAE Score", fontsize=16)
axes[0].set_xlabel("")
axes[0].tick_params(axis='x', labelsize=16)
axes[0].tick_params(axis='y', labelsize=16)
axes[0].legend()


sns.violinplot(ax=axes[1], x="Windowsize", y="MAE", data=df, inner="box", palette=palette_colors,saturation=1)
axes[1].set_title("Violin Plot of MAE for Different Windowsize", fontsize=16)
axes[1].set_ylabel("MAE Score", fontsize=16)
axes[1].set_xlabel("Windowsize", fontsize=16)
axes[1].tick_params(axis='x', labelsize=16)
axes[1].tick_params(axis='y', labelsize=16)

plt.tight_layout()
plt.savefig("SP_MAE.png", dpi=600, bbox_inches="tight")

# ===================
# Figure 25: R²
# ===================
fig, axes = plt.subplots(2, 1, figsize=(10, 6))

sns.boxplot(ax=axes[0], x="Windowsize", y="R2", data=df, palette=palette_colors,saturation=1)

means_r2 = df.groupby("Windowsize")["R2"].mean()
for i, (ws, mean_val) in enumerate(means_r2.items()):
    axes[0].scatter(i, mean_val, color="black", s=70, marker='^', zorder=3, label="Mean" if i == 0 else "")
    axes[0].text(i, mean_val + 0.005, f"{mean_val:.3f}", ha='center', va='bottom', fontsize=16)

axes[0].set_title("Boxplot of R² for Different Windowsize", fontsize=16)
axes[0].set_ylabel("R² Score", fontsize=16)
axes[0].set_xlabel("")
axes[0].tick_params(axis='x', labelsize=16)
axes[0].tick_params(axis='y', labelsize=16)
axes[0].legend()

sns.violinplot(ax=axes[1], x="Windowsize", y="R2", data=df, inner="box", palette=palette_colors,saturation=1)
axes[1].set_title("Violin Plot of R² for Different Windowsize", fontsize=16)
axes[1].set_ylabel("R² Score", fontsize=16)
axes[1].set_xlabel("Windowsize", fontsize=16)
axes[1].tick_params(axis='x', labelsize=16)
axes[1].tick_params(axis='y', labelsize=16)

plt.tight_layout()
plt.savefig("SP_R2.png", dpi=600, bbox_inches="tight")
"""




r"""
#figure 26 & 27
# ===================
# Figure_26 & 27
# ===================


file_path = r"C:\Users\zheng\Desktop\RUL_Battery\Experiment\different_fine_tuning\exp_result_FT.csv"
df = pd.read_csv(file_path, header=None, names=["FT_Ratio", "MAE", "RMSE", "R2"])
df["FT_Ratio"] = df["FT_Ratio"].astype(str)


plt.rcParams["font.family"] = "Times New Roman"


palette_colors = {
    "0.1": "#ff8c00",
    "0.4": "#c82423",
    "0.7": "#3c9d5d",
    "1.0": "#2878b5"
}

# ===================
# Figure 26: MAE
# ===================
fig, axes = plt.subplots(2, 1, figsize=(10, 6))


sns.boxplot(ax=axes[0], x="FT_Ratio", y="MAE", data=df, palette=palette_colors)

means = df.groupby("FT_Ratio")["MAE"].mean()
for i, (ratio, mean_val) in enumerate(means.items()):
    axes[0].scatter(i, mean_val, color="black", s=70, marker='^', zorder=3, label="Mean" if i == 0 else "")
    axes[0].text(i, mean_val + 0.005, f"{mean_val:.3f}", ha='center', va='bottom', fontsize=10)

axes[0].set_title("Boxplot of MAE for Different Fine-tuning Ratios", fontsize=12)
axes[0].set_ylabel("MAE Score", fontsize=14)
axes[0].set_xlabel("")
axes[0].tick_params(axis='x', labelsize=12)
axes[0].tick_params(axis='y', labelsize=12)
axes[0].legend()

# Violin Plot
sns.violinplot(ax=axes[1], x="FT_Ratio", y="MAE", data=df, inner="box", palette=palette_colors)
axes[1].set_title("Violin Plot of MAE for Different Fine-tuning Ratios", fontsize=12)
axes[1].set_ylabel("MAE Score", fontsize=14)
axes[1].set_xlabel("Fine-tuning Ratio", fontsize=14)
axes[1].tick_params(axis='x', labelsize=12)
axes[1].tick_params(axis='y', labelsize=12)

plt.tight_layout()
plt.savefig("Figure_26.png", dpi=600, bbox_inches="tight")

# ===================
# Figure 27: R²
# ===================
fig, axes = plt.subplots(2, 1, figsize=(10, 6))

sns.boxplot(ax=axes[0], x="FT_Ratio", y="R2", data=df, palette=palette_colors)

means_r2 = df.groupby("FT_Ratio")["R2"].mean()
for i, (ratio, mean_val) in enumerate(means_r2.items()):
    axes[0].scatter(i, mean_val, color="black", s=70, marker='^', zorder=3, label="Mean" if i == 0 else "")
    axes[0].text(i, mean_val + 0.005, f"{mean_val:.3f}", ha='center', va='bottom', fontsize=10)

axes[0].set_title("Boxplot of R² for Different Fine-tuning Ratios", fontsize=12)
axes[0].set_ylabel("R² Score", fontsize=14)
axes[0].set_xlabel("")
axes[0].tick_params(axis='x', labelsize=12)
axes[0].tick_params(axis='y', labelsize=12)
axes[0].legend()

sns.violinplot(ax=axes[1], x="FT_Ratio", y="R2", data=df, inner="box", palette=palette_colors)
axes[1].set_title("Violin Plot of R² for Different Fine-tuning Ratios", fontsize=12)
axes[1].set_ylabel("R² Score", fontsize=14)
axes[1].set_xlabel("Fine-tuning Ratio", fontsize=14)
axes[1].tick_params(axis='x', labelsize=12)
axes[1].tick_params(axis='y', labelsize=12)

plt.tight_layout()
plt.savefig("Figure_27.png", dpi=600, bbox_inches="tight")
"""





r"""
#===============================
#For presentation
#=================================
file_path = r"C:\Users\zheng\Desktop\RUL_Battery\Experiment\different_fine_tuning\exp_result_FT.csv"
df = pd.read_csv(file_path, header=None, names=["FT_Ratio", "MAE", "RMSE", "R2"])
df["FT_Ratio"] = df["FT_Ratio"].astype(str)

# 字体
plt.rcParams["font.family"] = "Segoe UI"


palette_colors = {
    "1.0": "#00549F",
    "0.7": "#7AB51D",
    "0.4": "#DA1F3D",
    "0.1": "#F29400"
}

# ===================
# Figure 26: MAE
# ===================
fig, axes = plt.subplots(2, 1, figsize=(10, 6))


sns.boxplot(ax=axes[0], x="FT_Ratio", y="MAE", data=df, palette=palette_colors,saturation=1)

means = df.groupby("FT_Ratio")["MAE"].mean()
for i, (ratio, mean_val) in enumerate(means.items()):
    axes[0].scatter(i, mean_val, color="black", s=70, marker='^', zorder=3, label="Mean" if i == 0 else "")
    axes[0].text(i, mean_val + 0.005, f"{mean_val:.3f}", ha='center', va='bottom', fontsize=16)

axes[0].set_title("Boxplot of MAE for Different Fine-tuning Ratios", fontsize=16)
axes[0].set_ylabel("MAE Score", fontsize=16)
axes[0].set_xlabel("")
axes[0].tick_params(axis='x', labelsize=16)
axes[0].tick_params(axis='y', labelsize=16)
axes[0].legend()

# Violin Plot
sns.violinplot(ax=axes[1], x="FT_Ratio", y="MAE", data=df, inner="box", palette=palette_colors,saturation=1)
axes[1].set_title("Violin Plot of MAE for Different Fine-tuning Ratios", fontsize=16)
axes[1].set_ylabel("MAE Score", fontsize=16)
axes[1].set_xlabel("Fine-tuning Ratio", fontsize=16)
axes[1].tick_params(axis='x', labelsize=16)
axes[1].tick_params(axis='y', labelsize=16)

plt.tight_layout()
plt.savefig("MAE_FT.png", dpi=600, bbox_inches="tight")

# ===================
# Figure 27: R²
# ===================
fig, axes = plt.subplots(2, 1, figsize=(10, 6))

sns.boxplot(ax=axes[0], x="FT_Ratio", y="R2", data=df, palette=palette_colors,saturation=1)

means_r2 = df.groupby("FT_Ratio")["R2"].mean()
for i, (ratio, mean_val) in enumerate(means_r2.items()):
    axes[0].scatter(i, mean_val, color="black", s=70, marker='^', zorder=3, label="Mean" if i == 0 else "")
    axes[0].text(i, mean_val + 0.005, f"{mean_val:.3f}", ha='center', va='bottom', fontsize=16)

axes[0].set_title("Boxplot of R² for Different Fine-tuning Ratios", fontsize=16)
axes[0].set_ylabel("R² Score", fontsize=16)
axes[0].set_xlabel("")
axes[0].tick_params(axis='x', labelsize=16)
axes[0].tick_params(axis='y', labelsize=16)
axes[0].legend()

sns.violinplot(ax=axes[1], x="FT_Ratio", y="R2", data=df, inner="box", palette=palette_colors)
axes[1].set_title("Violin Plot of R² for Different Fine-tuning Ratios", fontsize=16)
axes[1].set_ylabel("R² Score", fontsize=16)
axes[1].set_xlabel("Fine-tuning Ratio", fontsize=16)
axes[1].tick_params(axis='x', labelsize=16)
axes[1].tick_params(axis='y', labelsize=16)

plt.tight_layout()
plt.savefig("R2_FT.png", dpi=600, bbox_inches="tight")



# Figure 28 Average MAE and R Square from ablation Experiment
file_path_type1 = r"Experiment\Ablation Experiment\Type1\Type1_results_no_tf.csv"
df_type1 = pd.read_csv(file_path_type1)
file_path_type2 = r"Experiment\Ablation Experiment\Type2\Type2_results_no_tf.csv"
df_type2 = pd.read_csv(file_path_type2)
file_path_type3 = r"Experiment\Ablation Experiment\Type3\Type3_results_no_tf.csv"
df_type3 = pd.read_csv(file_path_type3)
file_path_type4 = r"Experiment\Ablation Experiment\Type4\Type4_results_no_tf.csv"
df_type4 = pd.read_csv(file_path_type4)



df_all = pd.concat([
    df_type1.assign(Experiment="Type1"),
    df_type2.assign(Experiment="Type2"),
    df_type3.assign(Experiment="Type3"),
    df_type4.assign(Experiment="Type4"),
])


plt.rcParams["font.family"] = "Times New Roman"


palette_colors = {
    "1": "#2878b5",
    "2": "#3c9d5d",
    "3": "#c82423",
    "4": "#ff8c00"
}
colors = [palette_colors["1"], palette_colors["2"],
          palette_colors["3"], palette_colors["4"]]


fig, axes = plt.subplots(2, 1, figsize=(7, 5))


sns.boxplot(ax=axes[0], x="Experiment", y="MAE",
            data=df_all, palette=colors)


mae_means = df_all.groupby("Experiment")["MAE"].mean()
for i, (exp, mean_val) in enumerate(mae_means.items()):
    axes[0].scatter(i, mean_val, color="black", s=70, marker="^", zorder=3, label="Mean" if i == 0 else "")
    axes[0].text(i, mean_val + 0.002, f"{mean_val:.3f}", 
                 ha='center', va='bottom', fontsize=10)

axes[0].set_title("Boxplot of MAE Across Seeds (Type1–Type4)")
axes[0].set_ylabel("MAE")
axes[0].set_xlabel("")
axes[0].tick_params(axis='both', labelsize=11)
axes[0].legend()


sns.boxplot(ax=axes[1], x="Experiment", y="R2",
            data=df_all, palette=colors)

r2_means = df_all.groupby("Experiment")["R2"].mean()
for i, (exp, mean_val) in enumerate(r2_means.items()):
    axes[1].scatter(i, mean_val, color="black", s=70, marker="^", zorder=3, label="Mean" if i == 0 else "")
    axes[1].text(i, mean_val + 0.002, f"{mean_val:.3f}", 
                 ha='center', va='bottom', fontsize=10)

axes[1].set_title("Boxplot of R² Across Seeds (Type1–Type4)")
axes[1].set_ylabel("R²")
axes[1].set_xlabel("Experiment Type")
axes[1].tick_params(axis='both', labelsize=11)
axes[1].legend()

plt.tight_layout()
plt.savefig("Figure_MAE_R2_Box.png", dpi=600, bbox_inches="tight")
plt.show()
"""

