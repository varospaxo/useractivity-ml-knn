import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('user_act_big_sorted.csv')

# Sort by prospect ID
sorted_df = df.sort_values(by='ProspectID')

# Calculate the size of each part
num_rows = sorted_df.shape[0]
part_size = num_rows // 6
remainder = num_rows % 6

# Split the DataFrame into 6 parts
dfs = []
start_idx = 0

for i in range(6):
    # Calculate the end index for this part
    end_idx = start_idx + part_size + (1 if i < remainder else 0)
    dfs.append(sorted_df.iloc[start_idx:end_idx])
    start_idx = end_idx

# Save each part to a separate CSV file
for i, df_part in enumerate(dfs):
    df_part.to_csv(f'parts/part_{i+1}.csv', index=False)

print("CSV file has been divided into 6 parts.")
