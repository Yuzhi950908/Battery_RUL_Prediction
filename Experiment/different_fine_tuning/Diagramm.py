
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 
file_path = r"C:\Users\zheng\Desktop\RUL_Battery\Experiment\different_fine_tuning\exp_result_FT.csv"

# 
df = pd.read_csv(file_path, header=None, names=["percentage", "MAE", "RMSE", "R2"])

# 
sns.set(style="whitegrid")

#  Boxplot
plt.figure(figsize=(10, 5))
sns.boxplot(x="percentage", y="MAE", data=df)
plt.title("Boxplot of MAE for Different Fine-Tuning Percentages")
plt.ylabel("MAE Score")
plt.xlabel("Fine-Tuning Percentage")
plt.tight_layout()
plt.show()

# Violin Plot
plt.figure(figsize=(10, 5))
sns.violinplot(x="percentage", y="MAE", data=df, inner="box")
plt.title("Violin Plot of MAE for Different Fine-Tuning Percentages")
plt.ylabel("MAE Score")
plt.xlabel("Fine-Tuning Percentage")
plt.tight_layout()
plt.show()
