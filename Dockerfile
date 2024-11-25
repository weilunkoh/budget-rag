# Use an official Miniconda image as the base image
FROM continuumio/miniconda3:latest

# Set the working directory in the container
WORKDIR /app

# add application code to /app
COPY . .

# Create the Conda environment
RUN conda env create -f conda-env-docker.yml

# Activate the Conda environment
SHELL ["conda", "run", "-n", "budget-rag", "/bin/bash", "-c"]

# Expose the port that the Python app will run on
EXPOSE 8501

# Specify the command to run on container start
CMD ["conda", "run", "--no-capture-output", "-n", "budget-rag", "streamlit", "run", "app.py"]

