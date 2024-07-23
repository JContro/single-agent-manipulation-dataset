# Set variables
PROJECT_ID="alpine-anvil-244411"
REPOSITORY_NAME="combined-app"
REGION="europe-west2"
IMAGE_NAME="combined-app"
TAG="latest"

# Ensure you're authenticated
gcloud auth login

# Configure Docker to use gcloud as a credential helper
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Set the correct project
gcloud config set project ${PROJECT_ID}

# Create the repository if it doesn't exist
gcloud artifacts repositories create ${REPOSITORY_NAME} \
    --repository-format=docker \
    --location=${REGION} \
    --description="Repository for combined-app"

# Tag the image for Artifact Registry
docker tag ${IMAGE_NAME}:${TAG} ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY_NAME}/${IMAGE_NAME}:${TAG}

# Push the image to Artifact Registry
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY_NAME}/${IMAGE_NAME}:${TAG}
