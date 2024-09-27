from google.cloud import storage
import json

def create_gcs_file_handler(bucket_name):
    """
    Creates and returns a function that handles opening files from a GCS bucket.
    :param bucket_name: Name of the GCS bucket
    :return: Function that opens and reads files from the specified GCS bucket
    """
    # Initialize GCS client using default credentials
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    def gcs_file_handler(blob_name):
        """
        Opens and reads a file from the GCS bucket.
        :param blob_name: Name of the blob (file) in the bucket
        :return: Parsed JSON content of the file
        """
        blob = bucket.blob(blob_name)
        content = blob.download_as_text()
        return json.loads(content)

    return gcs_file_handler

# Usage example:
if __name__ == "__main__":
    BUCKET_NAME = 'manipulation-dataset-kcl'

    # Create the file handler
    file_handler = create_gcs_file_handler(BUCKET_NAME)

    # Use the file handler to open different files
    manipulation_definitions = file_handler('manipulation-definitions.json')
    conversations = file_handler('conversations.json')
    human_responses = file_handler('human_responses.json')
    user_scores = file_handler('user_scores.json')
    user_timing = file_handler('user_timing.json')

    # Now you can work with the loaded data
    print("Manipulation Definitions:", manipulation_definitions)
    print("Conversations:", conversations)
    # ... and so on for other files