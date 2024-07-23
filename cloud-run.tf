# Configure the Google Cloud provider
provider "google" {
  project = "alpine-anvil-244411"
  region  = "europe-west2" # UK region
}

# Create a Cloud Run service
resource "google_cloud_run_service" "default" {
  name     = "cloudrun-srv"
  location = "europe-west2" # UK region

  template {
    spec {
      containers {
        image = "europe-west2-docker.pkg.dev/alpine-anvil-244411/combined-app/combined-app"
        ports {
          container_port = 8080
        }
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

# IAM entry for all users to invoke the function
resource "google_cloud_run_service_iam_member" "all_users" {
  service  = google_cloud_run_service.default.name
  location = google_cloud_run_service.default.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Display the service URL
output "service_url" {
  value = google_cloud_run_service.default.status[0].url
}
