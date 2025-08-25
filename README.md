# mothra-cams
hosting for mothra observatory cameras

## Camera Support

This system now supports both monochrome and color ZWO ASI cameras:
- **Monochrome cameras**: Standard mono cameras (e.g., ASI cameras without "MC" suffix)
- **Color cameras**: Color cameras like the ASI676MC for all-sky imaging

## Configuration

Camera configuration is handled via environment variables:

- `CAMERA_ID`: Camera index (default: 0)
- `CAMERA_NAME`: Name for the camera/mount (default: "b14m11")
- `CAMERA_TYPE`: "mono" or "color" (default: "mono")
- `SKIP_S3_UPLOAD`: Set to "true" to disable S3 upload for testing (default: "false")
- `DEBUG_MODE`: Set to "true" to disable continuous capture - camera initializes but waits for manual commands (default: "false")

## Docker Deployment

### Architecture Support

The system supports both x64 and ARM64 (Raspberry Pi) architectures:

- **x64 Linux**: Uses `ASI_linux_mac_SDK_V1.38/lib/x64/` libraries
- **ARM64 (Raspberry Pi 4)**: Uses `ASI_linux_mac_SDK_V1.38/lib/armv8/` libraries

### Standard Monochrome Camera

**x64 Linux:**
```bash
docker-compose up -d
# or explicitly:
ARCH=x64 docker-compose up -d
```

**Raspberry Pi:**
```bash
ARCH=armv8 docker-compose up -d
```

### Color All-Sky Camera (ASI676MC)

**x64 Linux:**
```bash
docker-compose -f docker-compose.allsky.yml up -d
```

**Raspberry Pi:**
```bash
ARCH=armv8 docker-compose -f docker-compose.allsky.yml up -d
```

### Testing Color Camera (No S3 Upload)

**Raspberry Pi (recommended):**
```bash
docker-compose -f docker-compose.test-rpi.yml up -d
```

**x64 Linux:**
```bash
docker-compose -f docker-compose.test-x64.yml up -d
```

## Testing

### Testing ASI676MC Color Camera in Container

**Important**: The test containers use `DEBUG_MODE=true` which prevents continuous imaging, allowing you to run manual tests without interference.

#### Architecture-Specific Testing

**Raspberry Pi 4 (ARM64):**
```bash
# Build and start the ARM64 container
docker-compose -f docker-compose.test-rpi.yml up -d

# Verify camera detection
docker logs asi676mc-test-rpi
```

**x64 Linux:**
```bash
# Build and start the x64 container  
docker-compose -f docker-compose.test-x64.yml up -d

# Verify camera detection
docker logs asi676mc-test-x64
```

**Note**: The system automatically selects the correct ASI SDK library (armv8 or x64) based on the Docker compose file used.
   
2. **Verify the container started and camera is detected**:
   ```bash
   # Check container logs
   docker logs asi676mc-test-macos  # or asi676mc-test
   ```
   
   The container will start in debug/idle mode and display:
   ```
   === RUNNING IN DEBUG/IDLE MODE ===
   Camera initialized and ready for manual testing.
   Continuous capture is DISABLED.
   Found 1 camera(s)
   Camera 0: ASI676MC
   ```

3. **Run the test script inside the container**:
   
   **Raspberry Pi:**
   ```bash
   # Basic test with default settings (0.01s exposure, gain 200)
   docker exec -it asi676mc-test-rpi python test_color_camera.py
   
   # Test with custom exposure and gain
   docker exec -it asi676mc-test-rpi python test_color_camera.py --exposure 2.0 --gain 300
   ```
   
   **x64 Linux:**
   ```bash
   # Basic test with default settings
   docker exec -it asi676mc-test-x64 python test_color_camera.py
   
   # Test with custom exposure and gain  
   docker exec -it asi676mc-test-x64 python test_color_camera.py --exposure 2.0 --gain 300
   ```

3. **Check the results**:
   - Images will be saved to your `/Users/ipasha/Documents/` folder
   - Look for files named `test-camera-YYYYMMDD_HHMMSS.webp` and `.png`

4. **Check container logs** (to see debug mode status):
   ```bash
   # Raspberry Pi
   docker logs asi676mc-test-rpi
   
   # x64 Linux
   docker logs asi676mc-test-x64
   ```

5. **Stop the test container**:
   ```bash
   # Raspberry Pi
   docker-compose -f docker-compose.test-rpi.yml down
   
   # x64 Linux
   docker-compose -f docker-compose.test-x64.yml down
   ```

### Debug Mode vs Production Mode

- **Debug Mode** (`DEBUG_MODE=true`): Camera initializes but doesn't capture continuously. Perfect for manual testing.
- **Production Mode** (`DEBUG_MODE=false`): Normal operation with continuous capture based on time/conditions.

For production deployment, always use `DEBUG_MODE=false` or omit the variable entirely.

### Test Script Options

- `--exposure, -e`: Exposure time in seconds (default: 0.01)
- `--gain, -g`: Camera gain 0-1000 (default: 200) 
- `--camera-type, -t`: "color" or "mono" (default: "color")

### Example Test Commands

```bash
# Quick test with short exposure
docker exec -it asi676mc-test python test_color_camera.py -e 0.01 -g 200

# Longer exposure for low light
docker exec -it asi676mc-test python test_color_camera.py -e 5.0 -g 400

# Test different gains
docker exec -it asi676mc-test python test_color_camera.py -e 1.0 -g 100
docker exec -it asi676mc-test python test_color_camera.py -e 1.0 -g 500
```
