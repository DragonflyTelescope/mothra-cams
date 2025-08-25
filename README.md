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

### Standard Monochrome Camera
```bash
docker-compose up -d
```

### Color All-Sky Camera (ASI676MC)
```bash
docker-compose -f docker-compose.allsky.yml up -d
```

### Testing Color Camera (No S3 Upload)
```bash
docker-compose -f docker-compose.test.yml up -d
```

## Testing

### Testing ASI676MC Color Camera in Container

**Important**: The test containers use `DEBUG_MODE=true` which prevents continuous imaging, allowing you to run manual tests without interference.

#### macOS USB Device Access

On macOS, USB device access in Docker requires special configuration:

1. **Start the test container** (macOS version with USB access):
   ```bash
   # Option 1: Use the macOS-optimized compose file
   docker-compose -f docker-compose.test-macos.yml up -d
   
   # Option 2: Use the standard test file (also macOS-compatible)
   docker-compose -f docker-compose.test-asi676mc.yml up -d
   ```
   
**Note**: The containers use `privileged: true` for USB access on macOS. If you prefer not to use privileged mode, you can try the alternative device mounting options commented in the compose files.
   
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
   ```bash
   # Basic test with default settings (0.01s exposure, gain 200)
   docker exec -it asi676mc-test python test_color_camera.py
   
   # Test with custom exposure and gain
   docker exec -it asi676mc-test python test_color_camera.py --exposure 2.0 --gain 300
   
   # Test mono mode
   docker exec -it asi676mc-test python test_color_camera.py --exposure 0.001 --gain 100 --camera-type mono
   ```

3. **Check the results**:
   - Images will be saved to your `/Users/ipasha/Documents/` folder
   - Look for files named `test-camera-YYYYMMDD_HHMMSS.webp` and `.png`

4. **Check container logs** (to see debug mode status):
   ```bash
   docker logs asi676mc-test
   ```

5. **Stop the test container**:
   ```bash
   docker-compose -f docker-compose.test-asi676mc.yml down
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
