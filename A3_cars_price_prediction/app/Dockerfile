# Use an official Python runtime as the base image
FROM python:3.11-slim

RUN pip install --upgrade pip
# Set the working directory inside the container
WORKDIR /root/app

COPY requirements.txt requirements.txt

# Install dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Expose Flask port
EXPOSE 5030

COPY code/ code/
WORKDIR /root/app/code

# Run Flask app
CMD ["python3", "app.py"]


# # Use a minimal Python runtime
# FROM python:3.11-slim

# # Set working directory
# WORKDIR /app

# # Copy application files
# COPY . /app

# # Install dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Expose Flask port
# EXPOSE 5000

# # Start the Flask application using Gunicorn
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
