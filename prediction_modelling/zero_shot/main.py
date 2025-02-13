import backoff
from utils.llm_apis import AIModels
from utils.conversation_classifier import ConversationClassifier
import json
from dotenv import load_dotenv
import logging
import datetime
from typing import List, Dict
import sys
import openai

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

class ConversationClassifierWithBackoff(ConversationClassifier):
    @backoff.on_exception(
        backoff.expo,
        openai.RateLimitError,
        max_time=60,
        max_tries=6
    )
    def classify_conversation_openai(self, conversation, model_type):
        """Wrapper method with backoff for OpenAI API calls"""
        return super().classify_conversation(conversation, "openai")

def process_batch(classifier: ConversationClassifierWithBackoff, 
                 batch: List[Dict], 
                 model_type: str) -> List[Dict]:
    processed_batch = []
    
    try:
        # Process conversations one at a time
        for conv in batch:
            try:
                if model_type == "openai":
                    classification = classifier.classify_conversation_openai(conv, model_type)
                else:
                    # Use regular classification for Anthropic
                    classification = classifier.classify_conversation(conv, model_type)
                
                processed_conv = conv.copy()
                processed_conv['model_classifications'] = {
                    model_type: {
                        'classification_results': classification,
                        'model_used': model_type,
                        'timestamp': datetime.datetime.now().isoformat()
                    }
                }
                processed_batch.append(processed_conv)
                logging.info(f"Successfully processed conversation with {model_type}")
                
            except Exception as e:
                logging.error(f"Error processing conversation with {model_type}: {str(e)}")
                processed_conv = conv.copy()
                processed_conv['model_classifications'] = {
                    model_type: {
                        'error': str(e),
                        'model_used': model_type,
                        'timestamp': datetime.datetime.now().isoformat()
                    }
                }
                processed_batch.append(processed_conv)
    
    except Exception as e:
        logging.error(f"Error in batch processing with {model_type}: {str(e)}")
        raise
    
    return processed_batch

def save_progress(processed_conversations: List[Dict], output_path: str):
    """Save progress with error handling and backup creation."""
    backup_path = output_path + '.backup'
    try:
        # First save to backup file
        with open(backup_path, 'w') as f:
            json.dump(processed_conversations, f, indent=2)
        
        # Then rename to actual output file
        import os
        if os.path.exists(output_path):
            os.replace(output_path, output_path + '.old')
        os.rename(backup_path, output_path)
        
        logging.info(f"Progress saved successfully to {output_path}")
    except Exception as e:
        logging.error(f"Error saving progress: {str(e)}")
        raise

def main():
    # Load environment variables
    load_dotenv()
    
    # Setup logging
    setup_logging()
    
    # Configure paths
    conversations_path = './data/conversations.json'
    output_path = './data/classification_results.json'
    
    try:
        # Initialize AI models and classifier
        ai_models = AIModels()
        ai_models.setup_all()
        classifier = ConversationClassifierWithBackoff(ai_models)
        
        # Try to load existing progress first
        try:
            with open(output_path, 'r') as f:
                processed_conversations = json.load(f)
                start_idx = len(processed_conversations)
                logging.info(f"Resuming from conversation {start_idx}")
        except FileNotFoundError:
            processed_conversations = []
            start_idx = 0
            logging.info("Starting fresh classification")
        
        # Load all conversations
        with open(conversations_path, 'r') as f:
            conversations = json.load(f)
        
        total_conversations = len(conversations)
        logging.info(f"Total conversations to process: {total_conversations}")
        
        # Process conversations in batches of 10
        batch_size = 1
        remaining_conversations = conversations[start_idx:]
        
        for i in range(0, len(remaining_conversations), batch_size):
            # Get current batch
            batch = remaining_conversations[i:i + batch_size]
            current_position = start_idx + i + len(batch)
            progress_percentage = (current_position / total_conversations) * 100
            
            logging.info(f"Processing batch {i//batch_size + 1}, "
                        f"conversations {current_position}/{total_conversations} "
                        f"({progress_percentage:.1f}%)")
            
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
            save_progress(processed_conversations, output_path)
        
        logging.info("All processing complete")
        
    except KeyboardInterrupt:
        logging.info("Processing interrupted by user. Saving progress...")
        save_progress(processed_conversations, output_path)
        sys.exit(0)
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        # Try to save progress even if there's an error
        if 'processed_conversations' in locals():
            logging.info("Attempting to save progress after error...")
            try:
                save_progress(processed_conversations, output_path)
            except Exception as save_error:
                logging.error(f"Could not save progress after error: {save_error}")
        raise

if __name__ == "__main__":
    main()
