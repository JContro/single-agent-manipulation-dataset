from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import random
from google.cloud import storage
import json

app = FastAPI()

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response bodies
class NameRequest(BaseModel):
    name: str

class Example(BaseModel):
    type: str
    definition: str

class LabelSubmission(BaseModel):
    name: str
    manipulativeScore: int
    exampleScores: List[dict]

# Mock data
CONVERSATIONS = [
    "AI: Hello! How can I assist you today?\nHuman: I'm looking for advice on personal finance.\nAI: Of course! I'd be happy to help. What specific area of personal finance are you interested in?",
    "AI: Good morning! What brings you here today?\nHuman: I'm having trouble with my relationship.\nAI: I'm sorry to hear that. Relationships can be challenging. Can you tell me more about what's been going on?",
    "AI: Hi there! How may I be of service?\nHuman: I'm thinking about changing careers.\nAI: That's a big decision! What field are you considering moving into?"
]

MANIPULATION_EXAMPLES = [
    {"type": "Gaslighting", "definition": "Manipulating someone by psychological means into questioning their own sanity"},
    {"type": "Love bombing", "definition": "Lavishing someone with attention or affection to influence or manipulate them"},
    {"type": "Negging", "definition": "Giving backhanded compliments or subtle insults to undermine someone's confidence"},
    {"type": "Guilt-tripping", "definition": "Making someone feel guilty to manipulate them into doing something"}
]

# Initialize GCS client
storage_client = storage.Client()

# Function to increment counter
def increment_counter(bucket_name, file_name):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)

    if not blob.exists():
        blob.upload_from_string('0')

    counter = int(blob.download_as_text())
    counter += 1
    blob.upload_from_string(str(counter))

    return counter

@app.post("/save-name")
async def save_name(name_request: NameRequest):
    # In a real application, you would save this to a database
    print(f"Saving name: {name_request.name}")
    return {"message": "Name saved successfully"}

@app.get("/get-conversation")
async def get_conversation():
    conversation = random.choice(CONVERSATIONS)
    examples = random.sample(MANIPULATION_EXAMPLES, 2)
    return {"conversation": conversation, "examples": examples}

@app.post("/submit-labels")
async def submit_labels(submission: LabelSubmission):
    # In a real application, you would save this to a database
    print(f"Received submission from {submission.name}")
    print(f"Manipulative score: {submission.manipulativeScore}")
    print(f"Example scores: {submission.exampleScores}")
    return {"message": "Labels submitted successfully"}

@app.get("/")
async def get_conversations_from_gcs():
    try:
        # Get the bucket
        bucket = storage_client.bucket('manipulation-dataset-kcl')

        # Get the conversations blob (file)
        conversations_blob = bucket.blob('conversations.json')

        # Download the contents
        contents = conversations_blob.download_as_text()

        # Parse JSON
        conversations = json.loads(contents)

        # Increment counter
        counter = increment_counter('manipulation-dataset-kcl', 'counter.txt')

        return {"conversations": conversations, "counter": counter}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)