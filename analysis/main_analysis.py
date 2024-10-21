from data_connection import create_gcs_file_handler 
from collections import defaultdict

BUCKET_NAME = 'manipulation-dataset-kcl'

file_handler = create_gcs_file_handler(BUCKET_NAME)

# Use the file handler to open different files
manipulation_definitions = file_handler('manipulation-definitions.json')
conversations = file_handler('conversations.json')
human_responses = file_handler('human_responses.json')
user_scores = file_handler('user_scores.json')
user_timing = file_handler('user_timing.json')

# print(human_responses[0])
# print(manipulation_definitions.keys())

# get the keys 
    # """
    # make something that looks like 
    #     conversation_id: {
    #         n_responses
    #         answers: 
    #             {
    #                 manip_type: [score]
    #             }
    #         }
    # """

def transform_responses(human_responses):
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
    
    return result


def check_conversation_completeness(transformed_data, required_manip_types):
    incomplete_conversations = {}
    for conv_id, conv_data in transformed_data.items():
        if conv_data['n_responses'] >= 3:
            missing_types = set(required_manip_types) - set(conv_data['answers'].keys())
            if missing_types:
                incomplete_conversations[conv_id] = list(missing_types)

    return incomplete_conversations

transformed_data = transform_responses(human_responses=human_responses)
print(check_conversation_completeness(transformed_data=transformed_data, required_manip_types=manipulation_definitions.keys()))


