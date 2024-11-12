import numpy as np
import pandas as pd
import krippendorff

def create_krippendorff_matrix(data):
    df = pd.DataFrame([{
        'email': d['email'],
        'conversation_id': d['conversation_id'],
        'score': d['scores']['General']
    } for d in data])
    
    # The key fix: we need to transpose the matrix to have raters as rows
    # and items as columns, as this is what krippendorff package expects
    matrix = pd.pivot(
        df,
        index='email',  # Changed from conversation_id
        columns='conversation_id',  # Changed from email
        values='score'
    ).to_numpy()
    
    return matrix

def transform_matrix(matrix):
    transformed = matrix.copy()
    transformed = np.where(transformed < 4, -1, transformed)
    transformed = np.where(transformed == 4, 0, transformed)
    transformed = np.where(transformed > 4, 1, transformed)
    return transformed

# Test data
perfect_agreement = [
    {'email': 'rater1@test.com', 'conversation_id': 'conv1', 'scores': {'General': 5}},
    {'email': 'rater2@test.com', 'conversation_id': 'conv1', 'scores': {'General': 5}},
    {'email': 'rater3@test.com', 'conversation_id': 'conv1', 'scores': {'General': 5}},
    {'email': 'rater1@test.com', 'conversation_id': 'conv2', 'scores': {'General': 2}},
    {'email': 'rater2@test.com', 'conversation_id': 'conv2', 'scores': {'General': 2}},
    {'email': 'rater3@test.com', 'conversation_id': 'conv2', 'scores': {'General': 2}},
]

complete_disagreement = [
    {'email': 'rater1@test.com', 'conversation_id': 'conv1', 'scores': {'General': 1}},
    {'email': 'rater2@test.com', 'conversation_id': 'conv1', 'scores': {'General': 3}},
    {'email': 'rater3@test.com', 'conversation_id': 'conv1', 'scores': {'General': 5}},
    {'email': 'rater1@test.com', 'conversation_id': 'conv2', 'scores': {'General': 5}},
    {'email': 'rater2@test.com', 'conversation_id': 'conv2', 'scores': {'General': 3}},
    {'email': 'rater3@test.com', 'conversation_id': 'conv2', 'scores': {'General': 1}},
]

known_pattern = [
    {'email': 'rater1@test.com', 'conversation_id': 'conv1', 'scores': {'General': 5}},
    {'email': 'rater2@test.com', 'conversation_id': 'conv1', 'scores': {'General': 5}},
    {'email': 'rater3@test.com', 'conversation_id': 'conv1', 'scores': {'General': 4}},
    {'email': 'rater1@test.com', 'conversation_id': 'conv2', 'scores': {'General': 2}},
    {'email': 'rater2@test.com', 'conversation_id': 'conv2', 'scores': {'General': 2}},
    {'email': 'rater3@test.com', 'conversation_id': 'conv2', 'scores': {'General': 3}},
]

test_cases = {
    "Perfect Agreement": perfect_agreement,
    "Complete Disagreement": complete_disagreement,
    "Known Pattern": known_pattern
}

for name, data in test_cases.items():
    print(f"\n{name}:")
    
    # Original matrix
    matrix = create_krippendorff_matrix(data)
    print("Original matrix (rows=raters, columns=items):")
    print(matrix)
    
    # Transformed matrix
    transformed = transform_matrix(matrix)
    print("\nTransformed matrix (-1, 0, 1):")
    print(transformed)
    
    # Calculate alpha for both
    alpha_original = krippendorff.alpha(reliability_data=matrix, level_of_measurement='ordinal')
    alpha_transformed = krippendorff.alpha(reliability_data=transformed, level_of_measurement='ordinal')
    
    print(f"\nOriginal alpha: {alpha_original:.3f}")
    print(f"Transformed alpha: {alpha_transformed:.3f}")