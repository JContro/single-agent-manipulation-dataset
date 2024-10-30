from datetime import datetime
import pytz

def remove_bad_responses(human_responses, user_timing):
    invalid_responses = find_bad_responses(user_timing)
    return [response for response in human_responses 
            if (response['email'], response['conversation_id']) not in invalid_responses]


def find_bad_responses(user_timing):
    invalid_responses = []
    
    for email in user_timing.keys():
        
        for conv_id, timing in user_timing[email].items():
            if 'submission_time' not in timing:
                continue
                
            request_time = datetime.fromisoformat(timing['request_time'])
            submission_time = datetime.fromisoformat(timing['submission_time'])
            time_delta = (submission_time - request_time).total_seconds()
            
            if time_delta <= 20:
                invalid_responses.append((email, conv_id))
                
    return  invalid_responses

def check_conversation_completeness(human_responses, required_manip_types):
    """
    Process conversation responses and check for completeness.
    
    Args:
        human_responses (list): List of response dictionaries containing conversation_id and scores
        required_manip_types (list): List of required manipulation types to check for
    
    Returns:
        tuple: (transformed_data dict, incomplete_conversations dict)
    """
    # Transform responses
    result = {}
    for response in human_responses:
        conv_id = response['conversation_id']
        if conv_id not in result:
            result[conv_id] = {"n_responses": 0, "answers": {}}
        
        result[conv_id]['n_responses'] += 1
        
        for manip_type, score in response['scores'].items():
            if manip_type not in result[conv_id]['answers']:
                result[conv_id]['answers'][manip_type] = []
            result[conv_id]['answers'][manip_type].append(score)
    
    # Check completeness
    incomplete_conversations = {}
    for conv_id, conv_data in result.items():
        if conv_data['n_responses'] >= 3:
            missing_types = set(required_manip_types) - set(conv_data['answers'].keys())
            if missing_types:
                incomplete_conversations[conv_id] = list(missing_types)
    
    return result, incomplete_conversations