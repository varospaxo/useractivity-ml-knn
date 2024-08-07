import pandas as pd

# Change the input csv file name with the required one
df = pd.read_csv('user_act.csv')

# Insert fieldname that needs to be sorted
sorted_df = df.sort_values(by='ProspectID')

# Change the output filename as required 
sorted_df.to_csv('user_act_big_prospect_id.csv', index=False)

print("CSV file has been sorted by ProspectID.")