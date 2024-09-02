# Use an official Python runtime as a parent image
FROM python:3.12-slim

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

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the Streamlit app
CMD ["poetry", "run", "streamlit", "run", "gui/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
