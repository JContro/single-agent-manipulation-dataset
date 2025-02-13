def validate_keys(data):
    """
    Validates that all required keys are present in the classification data.
    """
    def check_nested_keys(d, path):
        if not isinstance(d, dict):
            return []
        
        missing = []
        if path == '':
            # Check top level keys
            for model in ['openai', 'anthropic']:
                if model not in d:
                    missing.append(model)
                else:
                    # Check nested structure for each model
                    model_data = d[model]
                    missing.extend(check_nested_keys(model_data, model))
        else:
            # Check model-specific keys
            if 'classification_results' not in d:
                missing.append(f"{path}.classification_results")
            else:
                results = d['classification_results']
                if 'general' not in results:
                    missing.append(f"{path}.classification_results.general")
                if 'manipulation_tactics' not in results:
                    missing.append(f"{path}.classification_results.manipulation_tactics")
                else:
                    tactics = [
                        'Guilt-Tripping', 'Peer Pressure', 'Negging',
                        'Reciprocity Pressure', 'Gaslighting',
                        'Emotional Blackmail', 'Fear Enhancement'
                    ]
                    for tactic in tactics:
                        if tactic not in results['manipulation_tactics']:
                            missing.append(f"{path}.classification_results.manipulation_tactics.{tactic}")
            
            if 'model_used' not in d:
                missing.append(f"{path}.model_used")
            if 'timestamp' not in d:
                missing.append(f"{path}.timestamp")
        
        return missing

    missing = check_nested_keys(data, '')
    
    if missing:
        print("Warning: Missing the following keys:")
        for m in missing:
            print(f"  - {m}")
        raise KeyError(f"Missing keys: {', '.join(missing)}")
    
    return True