
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 
file_path = r"C:\Users\zheng\Desktop\RUL_Battery\Experiment\different_start_points\exp_result_SP.csv"

# 
df = pd.read_csv(file_path, header=None, names=["Windowsize", "MAE", "RMSE", "R2"])

# 
sns.set(style="whitegrid")

# 
plt.figure(figsize=(10, 5))
sns.boxplot(x="Windowsize", y="MAE", data=df)
plt.title("Boxplot of MAE for Different Windowsize")
plt.ylabel("MAE Score")
plt.xlabel("Windowsize")
plt.tight_layout()
plt.show()

# 
plt.figure(figsize=(10, 5))
sns.violinplot(x="Windowsize", y="MAE", data=df, inner="box")
plt.title("Violin Plot of MAE for Different Windowsize")
plt.ylabel("MAE Score")
plt.xlabel("Windowsize")
plt.tight_layout()
plt.show()


