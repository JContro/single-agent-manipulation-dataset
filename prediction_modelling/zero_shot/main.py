from utils.llm_apis import AIModels
from utils.conversation_classifier import ConversationClassifier
import json
from dotenv import load_dotenv
import logging
import datetime
from typing import List, Dict

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def process_batch(classifier: ConversationClassifier, 
                 batch: List[Dict], 
                 model_type: str) -> List[Dict]:
    processed_batch = []
    
    try:
        # Process all conversations in the batch using classify_conversation
        batch_results = [
            classifier.classify_conversation(conv, model_type)
            for conv in batch
        ]
        
        # Process results for each conversation in the batch
        for conv, classification in zip(batch, batch_results):
            processed_conv = conv.copy()
            processed_conv['model_classifications'] = {
                model_type: {
                    'classification_results': classification,
                    'model_used': model_type,
                    'timestamp': datetime.datetime.now().isoformat()
                }
            }
            processed_batch.append(processed_conv)
            
    except Exception as e:
        logging.error(f"Error processing batch with {model_type}: {str(e)}")
        # Handle failed batch by processing individually
        for conv in batch:
            processed_conv = conv.copy()
            processed_conv['model_classifications'] = {
                model_type: {
                    'error': str(e),
                    'model_used': model_type,
                    'timestamp': datetime.datetime.now().isoformat()
                }
            }
            processed_batch.append(processed_conv)
    
    return processed_batch

def main():
    # Load environment variables
    load_dotenv()
    
    # Setup logging
    setup_logging()
    
    try:
        # Initialize AI models and classifier
        ai_models = AIModels()
        ai_models.setup_all()
        classifier = ConversationClassifier(ai_models)
        
        # Load conversations
        conversations_path = './data/conversations.json'
        
        # Try to load existing progress first
        try:
            with open('./data/classification_results.json', 'r') as f:
                processed_conversations = json.load(f)
                start_idx = len(processed_conversations)
                logging.info(f"Resuming from conversation {start_idx}")
        except FileNotFoundError:
            processed_conversations = []
            start_idx = 0
            logging.info("Starting fresh")
        
        # Load all conversations
        with open(conversations_path, 'r') as f:
            conversations = json.load(f)
        
        # Limit number of conversations if specified
        num_conversations = 300
        if num_conversations is not None:
            conversations = conversations[:num_conversations]
        
        # Process conversations in batches of 10
        batch_size = 10
        remaining_conversations = conversations[start_idx:]
        
        for i in range(0, len(remaining_conversations), batch_size):
            
            # Get current batch
            batch = remaining_conversations[i:i + batch_size]
            logging.info(f"Processing batch {i//batch_size + 1}, conversations {i} to {min(i + batch_size, len(remaining_conversations))}")
            
            # Process batch for each model type
            batch_results = []
            for model_type in ["openai", "anthropic"]:
                processed_batch = process_batch(classifier, batch, model_type)
                
                # Merge results from different models
                for j, processed_conv in enumerate(processed_batch):
                    if len(batch_results) <= j:
                        batch_results.append(processed_conv)
                    else:
                        batch_results[j]['model_classifications'].update(
                            processed_conv['model_classifications']
                        )
            
            # Add processed batch to results
            processed_conversations.extend(batch_results)
            
            # Save progress after each batch
            with open('./data/classification_results.json', 'w') as f:
                json.dump(processed_conversations, f, indent=2)
            logging.info("Progress saved")
        
        logging.info("All processing complete")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()