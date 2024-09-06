# Use an official Python runtime as a parent image
ARG CUDA_IMAGE="12.1.1-devel-ubuntu22.04"
FROM nvidia/cuda:${CUDA_IMAGE}
# FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    python3-dev \
    python3-pip \
    build-essential \
    && apt-get clean

# Enable NVIDIA runtime for GPU access
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Set environment variables
ENV PYTHONPATH=/app

# Set the working directory in the container
WORKDIR /app

# Copy the Poetry files
COPY pyproject.toml poetry.lock* /app/

# Install Poetry
RUN pip install poetry

# Install dependencies
RUN poetry install --no-root --with api --without dev

# Install llama-cpp-python (build with cuda)
# RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python

# Copy the backend code into the container
COPY src/api /app/api
COPY src/data /app/data
COPY src/util /app/util

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the backend
CMD ["poetry", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
