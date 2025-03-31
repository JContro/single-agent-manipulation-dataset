import json 
import pandas as pd


fs = pd.read_csv('data/few_shot_items.csv')

fs = fs.reset_index() 
fs = fs.rename(columns={'Unnamed: 0': 'uuid'})

with open('data/classification_results.json','r') as f:
    convs = json.load(f)

co = pd.DataFrame(convs)

merged_df = pd.merge(co, fs, on='uuid', how='inner')

final_df = merged_df[['uuid','chat_completion','peer pressure_binary_true', 'reciprocity pressure_binary_true',
       'gaslighting_binary_true', 'guilt-tripping_binary_true',
       'emotional blackmail_binary_true', 'general_binary_true',
       'fear enhancement_binary_true', 'negging_binary_true']]

file_path = 'few_shot.json' 
json_data = final_df.to_json(orient='records')
with open(file_path, 'w') as f:
    f.write(json_data)

with open(file_path, 'r') as rf: 
    json_data = json.load(rf)

from pprint import pprint
for i in range(len(json_data)):
    print(f"--- Example {i} ---")
    pprint(json_data[i])