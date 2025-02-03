from get_analytics_df import get_analytics_df
import pandas as pd 
import json

df = get_analytics_df()

with open('data/conversation-contexts.json', 'r') as f:
    contexts = json.load(f)

cdf = pd.DataFrame(contexts)

# First, let's see which values don't match before joining
df_unmatched = df[~df['context'].isin(cdf['context'])]
cdf_unmatched = cdf[~cdf['context'].isin(df['context'])]

print("Unmatched values in df:")
print(df_unmatched['context'].unique())
print("\nUnmatched values in cdf:")
print(cdf_unmatched['context'].unique())

# Perform the inner join
merged_df = df.merge(cdf, on='context', how='inner')

# Print the number of rows before and after to see how many were matched
print(f"\nRows in df before merge: {len(df)}")
print(f"Rows in cdf before merge: {len(cdf)}")
print(f"Rows after inner merge: {len(merged_df)}")



print("\n=== Number of conversations by Category ===")
print(merged_df['category'].value_counts())

print("\n=== Number of conversations by Persuasion Strength ===")
print(merged_df['persuasion_strength'].value_counts(dropna=False))

print("\n=== Number of conversations by Manipulation Type ===")
print(merged_df['manipulation_type'].value_counts(dropna=False))

print("\n=== Number of conversations by Model ===")
print(merged_df['model'].value_counts())

# If you want percentages as well:
print("\n=== Percentages ===")
print("\nCategory percentages:")
print((merged_df['category'].value_counts(normalize=True) * 100).round(2), "%")

print("\nPersuasion Strength percentages:")
print((merged_df['persuasion_strength'].value_counts(normalize=True) * 100).round(2), "%")

print("\nManipulation Type percentages:")
print((merged_df['manipulation_type'].value_counts(normalize=True) * 100).round(2), "%")

print("\nModel percentages:")
print((merged_df['model'].value_counts(normalize=True) * 100).round(2), "%")


# set(merged_df['category'])
# {'Citizen Advice', 'Chatbot Conversation Topic', 'Personal Advice', 'Consumer Advice'}

# set(merged_df['persuasion_strength'])
# {'helpful', 'strong', None}

# set(merged_df['manipulation_type'])
# {'Peer Pressure', 'Guilt-Tripping', 'Reciprocity Pressure', 'Gaslighting', 'Negging', 'Fear Enhancement', 'Emotional Blackmail', None}

# set(merged_df['model'])
# {'gpt4', 'gemini', 'llama'}

# I want to get the number of conversations by each category, 