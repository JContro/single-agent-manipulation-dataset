from utils.llm_apis import AIModels
from utils.conversation_classifier import ConversationClassifier
import json
from dotenv import load_dotenv
import logging
import datetime

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

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
        num_conversations = 100
        if num_conversations is not None:
            conversations = conversations[:num_conversations]

        # Process remaining conversations
        for conversation in conversations[start_idx:]:
            import time
            logging.info("sleeping")
            time.sleep(60)
            
            logging.info(f"Processing conversation ID: {conversation.get('uuid', 'unknown')}")
            
            # Create a copy of the conversation data
            processed_conversation = conversation.copy()
            processed_conversation['model_classifications'] = {}

            # Classify using each model
            for model_type in ["openai", "anthropic"]:
                try:
                    classification = classifier.classify_conversation(conversation, model_type)
                    processed_conversation['model_classifications'][model_type] = {
                        'classification_results': classification,
                        'model_used': model_type,
                        'timestamp': datetime.datetime.now().isoformat()
                    }
                except Exception as e:
                    logging.error(f"Error processing with {model_type}: {str(e)}")
                    processed_conversation['model_classifications'][model_type] = {
                        'error': str(e),
                        'model_used': model_type,
                        'timestamp': datetime.datetime.now().isoformat()
                    }

            processed_conversations.append(processed_conversation)
            
            # Save progress after each conversation
            with open('./data/classification_results.json', 'w') as f:
                json.dump(processed_conversations, f, indent=2)
            logging.info("Progress saved")

        logging.info("All processing complete")

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()