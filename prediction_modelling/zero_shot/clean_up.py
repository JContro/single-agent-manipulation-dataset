import json 

with open('data/classification_results.json') as f:
    data = json.load(f)

def prepare_evaluation_data(classifications):
    """
    Prepare joined dataset of ground truth and model predictions.
    """
    rows = []
    for row in classifications:
        # Check if model_classifications exists
        if 'model_classifications' not in row:
            continue
            
        # Get model results, defaulting to empty dict if key doesn't exist
        openai_results = row['model_classifications'].get('openai', {})
        anthropic_results = row['model_classifications'].get('anthropic', {})
        
        # Check if classification_results exists and is not None for both
        if (openai_results.get('classification_results') is not None and 
            anthropic_results.get('classification_results') is not None):
            rows.append(row)
    
    return rows

cleaned_data = prepare_evaluation_data(data)


# Write filtered data to new file
with open('data/filtered_classification_results.json', 'w') as f:
    json.dump(cleaned_data, f, indent=2)
    


