# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Install Nginx and dos2unix
RUN apt-get update && apt-get install -y nginx dos2unix

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the nginx configuration file
COPY nginx.conf /etc/nginx/nginx.conf

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Copy the start script and convert to Unix line endings
COPY start.sh /start.sh
RUN dos2unix /start.sh && chmod +x /start.sh

# Run start.sh when the container launches
CMD ["/start.sh"]
