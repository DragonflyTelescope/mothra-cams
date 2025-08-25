FROM python:3.11-slim

# Build argument to specify architecture (x64 or armv8)
ARG ARCH=x64

# Install system dependencies
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    libusb-1.0-0-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (install zwoasi without auto-init)
RUN pip install pillow boto3 pytz astropy ephem numpy ghp-import requests \
    python-dotenv

# Install zwoasi without triggering auto-initialization
RUN pip install --no-deps zwoasi

# Create app directory
WORKDIR /app

# Copy the ZWO SDK for the specified architecture
# This will copy from either x64/ or armv8/ based on build arg
COPY ./ASI_linux_mac_SDK_V1.38/lib/${ARCH}/libASICamera2.so /usr/local/lib/
COPY ./ASI_linux_mac_SDK_V1.38/lib/${ARCH}/libASICamera2.so /usr/lib/
COPY ./ASI_linux_mac_SDK_V1.38/lib/${ARCH}/libASICamera2.so /app/

# Copy version-specific files if they exist (different versions for different archs)
COPY ./ASI_linux_mac_SDK_V1.38/lib/${ARCH}/libASICamera2.so.* /usr/local/lib/ 2>/dev/null || true

# Create symbolic links for different naming conventions
RUN ln -sf /usr/local/lib/libASICamera2.so /usr/local/lib/libASICamera2.so.1 || true
RUN ln -sf /usr/local/lib/libASICamera2.so /usr/local/lib/libASICamera2.so.1.38 || true
RUN ln -sf /usr/local/lib/libASICamera2.so /usr/local/lib/libASICamera2.so.1.39 || true

# Show what we copied for debugging
RUN echo "Architecture: ${ARCH}" && ls -la /usr/local/lib/libASI*

# Copy your Python files
COPY ./src/datetime_manager.py ./src/almanac.py ./src/capture.py /app/

# Create data directory
RUN mkdir -p /mothra/webcam/

# Update library cache and set environment variables for library loading
RUN ldconfig

# Set environment variables to help find the library
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/lib:/app
ENV ASI_LIB_PATH=/usr/local/lib/libASICamera2.so

ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "capture.py"]
