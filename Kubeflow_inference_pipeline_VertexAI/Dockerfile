# Start from official Python image
FROM python:3.10


# Install all needed libraries
RUN pip install \
    google-cloud-bigquery \
    google-cloud-storage \
    pandas \
    scikit-learn \
    db-dtypes \
    pyarrow \
    tensorflow


# (Optional) Set working directory
#WORKDIR /app

# (Optional) Copy your code inside container
# COPY . /app