# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all content from the repository to the working directory
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

# Expose port for Flask
EXPOSE 5000

# Command to run Flask app
CMD ["python", "app.py"]
