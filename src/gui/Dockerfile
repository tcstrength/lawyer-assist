# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y curl

# Set environment variables
ENV PYTHONPATH=/app

# Set the working directory in the container
WORKDIR /app

# Copy the Poetry files
COPY pyproject.toml poetry.lock* /app/

# Install Poetry
RUN pip install poetry

# Install dependencies
RUN poetry install --no-root --with frontend --without dev

# Copy the frontend (Streamlit) code into the container
COPY src/gui /app/gui
COPY src/util /app/util

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the Streamlit app
CMD ["poetry", "run", "streamlit", "run", "gui/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
