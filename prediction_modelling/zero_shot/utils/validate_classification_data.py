def validate_keys(data):
    expected_keys = [
        'openai',
        'anthropic',
        'openai.classification_results.manipulation_tactics.Guilt-Tripping',
        'openai.classification_results.manipulation_tactics.Peer Pressure',
        'openai.classification_results.manipulation_tactics.Negging',
        'openai.classification_results.manipulation_tactics.Reciprocity Pressure',
        'openai.classification_results.manipulation_tactics.Gaslighting',
        'openai.classification_results.manipulation_tactics.Emotional Blackmail',
        'openai.classification_results.manipulation_tactics.Fear Enhancement',
        'openai.classification_results.general',
        'openai.model_used',
        'openai.timestamp',
        'anthropic.classification_results.manipulation_tactics.Guilt-Tripping',
        'anthropic.classification_results.manipulation_tactics.Peer Pressure',
        'anthropic.classification_results.manipulation_tactics.Negging',
        'anthropic.classification_results.manipulation_tactics.Reciprocity Pressure',
        'anthropic.classification_results.manipulation_tactics.Gaslighting',
        'anthropic.classification_results.manipulation_tactics.Emotional Blackmail',
        'anthropic.classification_results.manipulation_tactics.Fear Enhancement',
        'anthropic.classification_results.general',
        'anthropic.model_used',
        'anthropic.timestamp'
    ]
    
    def get_value(d, path):
        keys = path.split('.')
        current = d
        for key in keys:
            if key not in current:
                return None
            current = current[key]
        return current

    missing = [key for key in expected_keys if get_value(data, key) is None]
    
    if missing:
        raise KeyError(f"Missing keys: {', '.join(missing)}")
    return True