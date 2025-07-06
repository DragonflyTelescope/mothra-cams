FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    libusb-1.0-0-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install zwoasi pillow boto3 pytz astropy ephem numpy ghp-import \
    python-dotenv

# Create app directory
WORKDIR /app

# Copy the ZWO SDK first (before trying to use it)
COPY ./ASI_linux_mac_SDK_V1.38/lib/x64/libASICamera2.so /usr/local/lib/

# Copy your Python files
COPY ./src/datetime_manager.py ./src/almanac.py ./src/capture.py /app/

# Create data directory
RUN mkdir -p /mothra/webcam/

# Update library cache
RUN ldconfig

ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "capture.py"]
