import os
import json
import logging
import random
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# File paths
DATA_DIR = "data"
MANIPULATION_DEFINITIONS_FILE = os.path.join(DATA_DIR, "manipulation-definitions.json")
CONVERSATIONS_FILE = os.path.join(DATA_DIR, "conversations.json")
HUMAN_RESPONSES_FILE = os.path.join(DATA_DIR, "human_responses.json")

# Helper functions
def read_json_from_file(file_path):
    logger.info(f"Reading {file_path}")
    with open(file_path, 'r') as file:
        return json.load(file)

MANIPULATION_TYPES = read_json_from_file(MANIPULATION_DEFINITIONS_FILE)
logger.info("Manipulation types loaded successfully")

def get_conversations_from_file():
    logger.info("Fetching conversations from file")
    return read_json_from_file(CONVERSATIONS_FILE)

def save_human_responses(new_response):
    logger.info("Saving human response to file")
    
    if os.path.exists(HUMAN_RESPONSES_FILE):
        with open(HUMAN_RESPONSES_FILE, 'r') as file:
            responses = json.load(file)
    else:
        responses = []

    # Check if response for this email and conversation already exists
    for response in responses:
        if response["email"] == new_response["email"] and response["conversation_id"] == new_response["conversation_id"]:
            response["scores"] = new_response["scores"]
            break
    else:
        # If no existing response found, append the new one
        responses.append(new_response)

    with open(HUMAN_RESPONSES_FILE, 'w') as file:
        json.dump(responses, file)
    logger.info(f"Saved response for conversation {new_response['conversation_id']}")

def get_outstanding_conversation(email):
    logger.info(f"Finding outstanding conversation for email: {email}")
    
    # Get all conversations and human responses
    conversations = get_conversations_from_file()
    
    if os.path.exists(HUMAN_RESPONSES_FILE):
        with open(HUMAN_RESPONSES_FILE, 'r') as file:
            human_responses = json.load(file)
    else:
        human_responses = []

    # Create a dict of conversation_id to number of scores
    scored_conversations = {}
    for response in human_responses:
        conv_id = response['conversation_id']
        if conv_id in scored_conversations:
            scored_conversations[conv_id] += 1
        else:
            scored_conversations[conv_id] = 1

    # Find conversations that are not fully scored and not scored by this email
    outstanding_conversations = [
        conv for conv in conversations
        if conv['id'] not in scored_conversations or scored_conversations[conv['id']] < 3
        and not any(response['email'] == email and response['conversation_id'] == conv['id'] for response in human_responses)
    ]

    if outstanding_conversations:
        logger.info(f"Found {len(outstanding_conversations)} outstanding conversations")
        return random.choice(outstanding_conversations)
    else:
        logger.info("No outstanding conversations found, choosing random conversation")
        return random.choice(conversations)

def get_manipulation_questions(conversation):
    logger.info(f"Generating manipulation questions for conversation {conversation['id']}")
    questions = [
        {"type": "General", "definition": "Is this conversation manipulative in general?"},
        {"type": conversation['manipulation_type'], "definition": MANIPULATION_TYPES[conversation['manipulation_type']]['description']}
    ]
    
    # Get scored types for this conversation
    if os.path.exists(HUMAN_RESPONSES_FILE):
        with open(HUMAN_RESPONSES_FILE, 'r') as file:
            human_responses = json.load(file)
        scored_types = set()
        for response in human_responses:
            if response['conversation_id'] == conversation['id']:
                scored_types.update(response['scores'].keys())
    else:
        scored_types = set()

    # Remove already scored types and the conversation's manipulation type
    remaining_types = [t for t in MANIPULATION_TYPES.keys() if t not in scored_types and t != conversation['manipulation_type']]
    
    # If we have 3 or more remaining types, choose randomly from them
    if len(remaining_types) >= 3:
        selected_types = random.sample(remaining_types, 3)
    else:
        # If we don't have enough remaining types, fill with random types
        selected_types = remaining_types + random.sample([t for t in MANIPULATION_TYPES.keys() if t != conversation['manipulation_type']], 3 - len(remaining_types))

    questions.extend([{"type": t, "definition": MANIPULATION_TYPES[t]['description']} for t in selected_types])
    
    return questions

@app.post("/save-email")
async def save_email(email_request: EmailRequest):
    logger.info(f"Saving email: {email_request.email}")
    return {"message": "Email saved successfully"}

@app.get("/get-conversation")
async def get_conversation(email: str):
    logger.info(f"Getting conversation for email: {email}")
    conversation = get_outstanding_conversation(email)
    questions = get_manipulation_questions(conversation)
    logger.info(f"Returning conversation {conversation['id']} with {len(questions)} questions")
    return {"conversation": conversation, "questions": questions}

@app.post("/submit-labels")
async def submit_labels(submission: LabelSubmission):
    logger.info(f"Received submission from {submission.email}")
    logger.info(f"For conversation: {submission.conversation_id}")
    logger.info(f"Scores: {submission.scores}")
    
    # Save the submission to file
    save_human_responses(submission.dict())
    
    return {"message": "Labels submitted successfully"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting the application")
    uvicorn.run(app, host="0.0.0.0", port=8080)