import json
import logging
from datetime import datetime, timezone
from typing import Dict

import random
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI()

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class EmailRequest(BaseModel):
    email: str

class LabelSubmission(BaseModel):
    email: str
    conversation_id: str
    scores: Dict[str, int]

# GCS configuration
BUCKET_NAME = 'manipulation-dataset-kcl'
BLOB_NAMES = {
    'manipulation_definitions': 'manipulation-definitions.json',
    'conversations': 'conversations.json',
    'human_responses': 'human_responses.json',
    'user_scores': 'user_scores.json',
    'timing': 'user_timing.json'
}

# Initialize GCS client
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

# Helper functions
def read_json_from_gcs(blob_name: str) -> dict:
    """Read JSON data from a GCS blob."""
    logger.info(f"Reading {blob_name} from GCS bucket {BUCKET_NAME}")
    blob = bucket.blob(blob_name)
    content = blob.download_as_text()
    return json.loads(content)

def write_json_to_gcs(blob_name: str, data: dict):
    """Write JSON data to a GCS blob."""
    logger.info(f"Writing to {blob_name} in GCS bucket {BUCKET_NAME}")
    blob = bucket.blob(blob_name)
    blob.upload_from_string(json.dumps(data))

# Load manipulation types
MANIPULATION_TYPES = read_json_from_gcs(BLOB_NAMES['manipulation_definitions'])
logger.info("Manipulation types loaded successfully")

def get_conversations_from_gcs() -> list:
    """Fetch conversations from GCS."""
    logger.info("Fetching conversations from GCS")
    return read_json_from_gcs(BLOB_NAMES['conversations'])

def save_human_responses(new_response: dict):
    """Save human response to GCS."""
    logger.info("Saving human response to GCS")
    
    blob = bucket.blob(BLOB_NAMES['human_responses'])
    responses = json.loads(blob.download_as_text()) if blob.exists() else []

    # Update existing response or append new one
    for response in responses:
        if response["email"] == new_response["email"] and response["conversation_id"] == new_response["conversation_id"]:
            response["scores"] = new_response["scores"]
            break
    else:
        responses.append(new_response)

    write_json_to_gcs(BLOB_NAMES['human_responses'], responses)
    logger.info(f"Saved response for conversation {new_response['conversation_id']}")

def save_timing_info(email: str, conversation_id: str, request_time: str = None, submission_time: str = None):
    """Save timing information to GCS."""
    logger.info(f"Saving timing info for email: {email}, conversation: {conversation_id}")
    blob = bucket.blob(BLOB_NAMES['timing'])
    timing_info = json.loads(blob.download_as_text()) if blob.exists() else {}

    if email not in timing_info:
        timing_info[email] = {}

    if conversation_id not in timing_info[email]:
        timing_info[email][conversation_id] = {}

    if request_time:
        timing_info[email][conversation_id]["request_time"] = request_time
    if submission_time:
        timing_info[email][conversation_id]["submission_time"] = submission_time

    # Calculate and save total time if both request_time and submission_time are provided
    if request_time and submission_time:
        logger.info(f"Request time: {request_time}, Submission time: {submission_time}")
        request_datetime = datetime.fromisoformat(request_time)
        submission_datetime = datetime.fromisoformat(submission_time)
        total_time = (submission_datetime - request_datetime).total_seconds()
        timing_info[email][conversation_id]["total_time"] = total_time
    write_json_to_gcs(BLOB_NAMES['timing'], timing_info)

def get_outstanding_conversation(email: str) -> dict:
    """Find an outstanding conversation for the given email."""
    logger.info(f"Finding outstanding conversation for email: {email}")
    
    conversations = get_conversations_from_gcs()
    human_responses = read_json_from_gcs(BLOB_NAMES['human_responses']) if bucket.blob(BLOB_NAMES['human_responses']).exists() else []

    scored_conversations = {}
    for response in human_responses:
        
        conv_id = response['conversation_id']
        scored_conversations[conv_id] = scored_conversations.get(conv_id, 0) + 1

    outstanding_conversations = [
        conv for conv in conversations
        if (conv['uuid'] not in scored_conversations or scored_conversations[conv['uuid']] < 3) and
        not any(response['email'] == email and response['conversation_id'] == conv['uuid'] for response in human_responses)
    ]

    if outstanding_conversations:
        logger.info(f"Found {len(outstanding_conversations)} outstanding conversations")
        return random.choice(outstanding_conversations)
    else:
        logger.info("No outstanding conversations found, choosing random conversation")
        return random.choice(conversations)

# def get_manipulation_questions(conversation: dict) -> list:
#     """Generate manipulation questions for a given conversation."""
#     logger.info(f"Generating manipulation questions for conversation {conversation['uuid']}")
#     questions = [{"type": "General", "definition": "The AI agent was manipulative."}]
    
#     human_responses = read_json_from_gcs(BLOB_NAMES['human_responses']) if bucket.blob(BLOB_NAMES['human_responses']).exists() else []
    
#     scored_types = set()
#     for response in human_responses:
#         if response['conversation_id'] == conversation['uuid']:
#             for score_type in response['scores']:
#                 scored_types.add(score_type)
    
#     remaining_types = []
#     for t in MANIPULATION_TYPES.keys():
#         if t not in scored_types:
#             remaining_types.append(t)

#     """
#     if it is manipulation type
#     choose the manipulation type + 2 from the remaining types

#     if it is persuasion then randomly pick 3 questions from the remaining types
#     """
#     questions.append({"type": conversation['manipulation_type'], "definition": MANIPULATION_TYPES[conversation['manipulation_type']]['description']})


#     selected_types = (random.sample(remaining_types, 3) if len(remaining_types) >= 3 
#                       else remaining_types + random.sample([t for t in MANIPULATION_TYPES.keys() if t != conversation['manipulation_type']], 3 - len(remaining_types)))

#     questions.extend([{"type": t, "definition": MANIPULATION_TYPES[t]['description']} for t in selected_types])
    
#     return questions[:1] + random.sample(questions[1:], len(questions) - 1)

def get_manipulation_questions(conversation: dict) -> list:
    """Generate manipulation questions for a given conversation."""
    logger.info(f"Generating manipulation questions for conversation {conversation['uuid']}")

    # Initialize questions with a general manipulation question
    questions = [{"type": "General", "definition": "The AI agent was manipulative."}]

    # Read human responses from GCS or use an empty list if not found
    human_responses = read_json_from_gcs(BLOB_NAMES['human_responses']) if bucket.blob(BLOB_NAMES['human_responses']).exists() else []

    # Get scored types for the current conversation
    scored_types = set()
    for response in human_responses:
        if response['conversation_id'] == conversation['uuid']:
            scored_types.update(response['scores'])

    # Get remaining types that haven't been scored
    remaining_types = [t for t in MANIPULATION_TYPES.keys() if t not in scored_types and t != conversation['manipulation_type']]
    if conversation['prompt_type'] == 'manipulation': 
        questions.append({
            "type": conversation['manipulation_type'],
            "definition": MANIPULATION_TYPES[conversation['manipulation_type']]['description']
        })
        # For other types, choose 2 from remaining types
        if len(remaining_types) >= 2:
            selected_types = random.sample(remaining_types, 2)
        else:
            # If less than 2 remaining types, use all remaining types and sample from all types to get 2
            selected_types = remaining_types + random.sample(
                [t for t in MANIPULATION_TYPES.keys() if t != conversation['manipulation_type'] and t not in remaining_types],
                2 - len(remaining_types)
            )
    elif conversation['prompt_type'] == 'persuasion': 
        # For Persuasion, randomly pick 3 questions from remaining types
        if len(remaining_types) >= 3:
            selected_types = random.sample(remaining_types, 3)
        else:
            # If less than 3 remaining types, use all remaining types and sample from all types to get 3
            selected_types = remaining_types + random.sample(
                [t for t in MANIPULATION_TYPES.keys() if t != conversation['manipulation_type'] and t not in remaining_types],
                3 - len(remaining_types)
            )
    else:
        raise ValueError("There was a data incosistency error")


    # Add selected types to questions
    for t in selected_types:
        questions.append({
            "type": t,
            "definition": MANIPULATION_TYPES[t]['description']
        })

    # Randomize the order of questions (except the first one)
    return questions[:1] + random.sample(questions[1:], len(questions) - 1)

def get_user_scores() -> dict:
    """Fetch user scores from GCS."""
    logger.info("Fetching user scores from GCS")
    blob = bucket.blob(BLOB_NAMES['user_scores'])
    return json.loads(blob.download_as_text()) if blob.exists() else {}

def update_user_scores(email: str):
    """Update score count for a user."""
    logger.info(f"Updating score count for user: {email}")
    user_scores = get_user_scores()
    user_scores[email] = user_scores.get(email, 0) + 1
    write_json_to_gcs(BLOB_NAMES['user_scores'], user_scores)

# API Endpoints
@app.post("/save-email")
async def save_email(email_request: EmailRequest):
    logger.info(f"Saving email: {email_request.email}")
    return {"message": "Email saved successfully"}

@app.get("/get-conversation")
async def get_conversation(email: str):
    logger.info(f"Getting conversation for email: {email}")
    conversation = get_outstanding_conversation(email)
    questions = get_manipulation_questions(conversation)
    
    request_time = datetime.now(timezone.utc).isoformat()
    save_timing_info(email, conversation['uuid'], request_time=request_time)
    
    logger.info(f"Returning conversation {conversation['uuid']} with {len(questions)} questions")
    return {"conversation": conversation, "questions": questions}

@app.post("/submit-labels")
async def submit_labels(submission: LabelSubmission):
    logger.info(f"Received submission from {submission.email} for conversation: {submission.conversation_id}")
    logger.info(f"Scores: {submission.scores}")
    
    save_human_responses(submission.dict())
    update_user_scores(submission.email)
    
    submission_time = datetime.now(timezone.utc).isoformat()
    save_timing_info(submission.email, submission.conversation_id, submission_time=submission_time)
    
    return {"message": "Labels submitted successfully"}

@app.get("/get-scored-conversations")
async def get_scored_conversations(email: str):
    logger.info(f"Getting scored conversations count for email: {email}")
    user_scores = get_user_scores()
    count = user_scores.get(email, 0)
    logger.info(f"User {email} has scored {count} conversations")
    return {"email": email, "scored_conversations": count}

@app.get("/get-timing-info")
async def get_timing_info(email: str = None):
    logger.info(f"Getting timing info for email: {email}")
    blob = bucket.blob(BLOB_NAMES['timing'])
    if blob.exists():
        timing_info = json.loads(blob.download_as_text())
        return {"email": email, "timing_info": timing_info.get(email, {})} if email else timing_info
    else:
        return {}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting the application")
    uvicorn.run(app, host="0.0.0.0", port=8080)