import pandas as pd
from scipy.stats import ttest_ind

'''
 @Author: Gagné-Labelle, Guillaume & Finoude, Meriadec 
 @Student number: 20174375 & B9592
 @Date: April, 2023
 @Project: Rentabilisation énergétique des réseaux de neurones - IFT3710 - UdeM
'''

combined_file = 'combined.csv'

# Read the combined file into a pandas dataframe
df = pd.read_csv(combined_file)

# Remove the "_seedX.csv" extension from the Filename column
df['Filename'] = df['Filename'].apply(lambda x: x[:-10])

df.rename(columns={'Filename': 'Group'}, inplace=True)

# Split the dataframe into two parts
df_cpu = df[df['Group'].str.startswith('cpu')]
df_gpu = df[df['Group'].str.startswith('gpu')]

# Define the three groups based on the Group column
group_cpu_0x1 = df_cpu.loc[df_cpu['Group'] == 'cpu_inf_pp0x1', 'emissions']
group_cpu_68x3 = df_cpu.loc[df_cpu['Group'] == 'cpu_inf_pp68x3', 'emissions']
group_cpu_90x2 = df_cpu.loc[df_cpu['Group'] == 'cpu_inf_pp90x2', 'emissions']

group_gpu_0x1 = df_gpu.loc[df_gpu['Group'] == 'gpu_inf_pp0x1', 'emissions']
group_gpu_68x3 = df_gpu.loc[df_gpu['Group'] == 'gpu_inf_pp68x3', 'emissions']
group_gpu_90x2 = df_gpu.loc[df_gpu['Group'] == 'gpu_inf_pp90x2', 'emissions']

# Perform t-test for group_68x3 vs. group_0x1
cpu_tvalue1, cpu_pvalue1 = ttest_ind(group_cpu_68x3, group_cpu_0x1)
gpu_tvalue1, gpu_pvalue1 = ttest_ind(group_gpu_68x3, group_gpu_0x1)

# Perform t-test for group_90x2 vs. group_0x1
cpu_tvalue2, cpu_pvalue2 = ttest_ind(group_cpu_90x2, group_cpu_0x1)
gpu_tvalue2, gpu_pvalue2 = ttest_ind(group_gpu_90x2, group_gpu_0x1)


#Print the results
print(f"T-test for CPU emissions:")
print(f"Group cpu_inf_pp68x3 vs. cpu_inf_pp0x1: t-value = {cpu_tvalue1:.3f}, p-value = {cpu_pvalue1:.6f}")
print(f"Group cpu_inf_pp90x2 vs. cpu_inf_pp0x1: t-value = {cpu_tvalue2:.3f}, p-value = {cpu_pvalue2:.6f}")

print("================================")

print(f"T-test for GPU emissions:")
print(f"Group gpu_inf_pp68x3 vs. gpu_inf_pp0x1: t-value = {gpu_tvalue1:.3f}, p-value = {gpu_pvalue1:.6f}")
print(f"Group gpu_inf_pp90x2 vs. gpu_inf_pp0x1: t-value = {gpu_tvalue2:.3f}, p-value = {gpu_pvalue2:.6f}")
