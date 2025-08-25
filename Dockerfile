FROM python:3.11-slim

# Build argument to specify architecture (x64 or armv8)
ARG ARCH=x64

# Install system dependencies
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    libusb-1.0-0-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install pillow boto3 pytz astropy ephem numpy ghp-import requests \
    python-dotenv
RUN pip install --no-deps zwoasi

# Create app directory
WORKDIR /app

# Copy the ZWO SDK files for the specified architecture
COPY ./ASI_linux_mac_SDK_V1.38/lib/${ARCH}/libASICamera2.so* /usr/local/lib/
COPY ./ASI_linux_mac_SDK_V1.38/lib/${ARCH}/libASICamera2.a /usr/local/lib/

# Update library cache
RUN ldconfig

# Copy your Python files
COPY ./src/ /app/
COPY ./*.py /app/

# Set environment for better debugging
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "capture.py"]