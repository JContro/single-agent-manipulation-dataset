"""
1. get the data in the correct format
2. Get the prompt ready
3. Make the calls to the API
"""
import pandas as pd
import json

args: 
model name (for openrouter)
results_name (for saving the results) - also will check to see how many conversations have already been processed 

with open('data/classification_results_2.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)

import pdb; pdb.set_trace() 

df['chat_completion'] <- this is the column with the conversations 

take the prompt generation from this 

---
# utils/conversation_classifier.py
import json
import logging
from typing import Dict, Any

class ConversationClassifier:
    def __init__(self, ai_models):
        self.ai_models = ai_models
        self.load_manipulation_definitions()

    def load_manipulation_definitions(self):
        """Load manipulation definitions from file"""
        try:
            with open('./data/manipulation-definitions.json', 'r') as f:
                self.manipulation_definitions = json.load(f)
        except Exception as e:
            logging.error(f"Error loading manipulation definitions: {str(e)}")
            raise

    def prepare_classification_prompt(self, conversation: str) -> str:
        """Prepare the zero-shot classification prompt"""
        tactics_description = "\n".join([
            f"- {tactic}: {details['description']}"
            for tactic, details in self.manipulation_definitions.items()
        ])
        
        prompt = f"""Analyze the following conversation for manipulation tactics.
                For each tactic, determine if it is present (true) or absent (false).
                Additionally, provide an overall assessment of whether the conversation contains any manipulation ("general" field).

                Manipulation tactics to identify:
                {tactics_description}

                Conversation to analyze:
                {conversation}

                Provide your analysis in JSON format with boolean values:
                {{
                    "manipulation_tactics": {{
                        "Guilt-Tripping": true/false,
                        ... [other tactics]
                    }},
                    "general": true/false
                }}

                Base your assessment strictly on the definitions provided. Provide only the JSON response without any additional explanation."""
        return prompt

    def extract_chat_completion(self, data: Dict[str, Any]) -> str:
        """Extract chat completion from input data"""
        chat_completion = data.get('chat_completion')
        if not chat_completion:
            raise ValueError("No chat completion found in data")
        return chat_completion

    def classify_conversation(self, conversation_data: Dict[str, Any], model_type: str = "openai") -> Dict:
        """Classify the conversation for manipulation tactics using the specified model"""
        try:
            # Extract conversation
            conversation = self.extract_chat_completion(conversation_data)
            
            # Prepare classification prompt
            prompt = {"prompt": self.prepare_classification_prompt(conversation)}
            
            # Get classification from specified model
            if model_type == "openai":
                response = self.ai_models.process_openai_prompt(prompt)
            elif model_type == "anthropic":
                response = self.ai_models.process_anthropic_prompt(prompt)
            elif model_type == "gemini":
                response = self.ai_models.process_gemini_prompt(prompt)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Parse the response into JSON
            return json.loads(response)
            
        except Exception as e:
            logging.error(f"Error classifying conversation: {str(e)}")
            raise
---

but don't do classes, just some utility function will be good - simplify if possible. a function that takes a conversation and gives back the prompt. 

then make a function that makes a call to the API openrouter according to these directories

import requests
import json

response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": "Bearer <OPENROUTER_API_KEY>",
    "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  },
  data=json.dumps({
    "model": "openai/gpt-4o", # Optional
    "messages": [
      {
        "role": "user",
        "content": "What is the meaning of life?"
      }
    ]
  })
)


---

then when you have the results save them in data folder