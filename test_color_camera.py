#!/usr/bin/env python3
"""
Test script for ASI cameras (color and mono)
This script will capture a single image and save it to /tmp/ inside the container.
Designed to be run inside the Docker container with: docker exec -it <container> python test_color_camera.py

Usage:
  python test_color_camera.py [--exposure SECONDS] [--gain VALUE] [--camera-type TYPE]

Examples:
  python test_color_camera.py --exposure 0.01 --gain 200
  python test_color_camera.py --exposure 2.0 --gain 300 --camera-type color
  python test_color_camera.py --exposure 0.001 --gain 100 --camera-type mono
"""

import argparse
import os
import sys

import zwoasi as asi

from capture import ObservatoryCamera, init_asi_library


def test_camera(exposure_time=0.01, gain=200, camera_type="color"):
    """Test the camera functionality"""

    # Initialize the ASI library using the same function as capture.py
    try:
        init_asi_library()
    except Exception as e:
        print(f"Failed to initialize ASI library: {e}")
        print("Make sure the ASI SDK is properly installed")
        return False

    # Check if any cameras are connected
    num_cameras = asi.get_num_cameras()
    print(f"Found {num_cameras} camera(s)")

    if num_cameras == 0:
        print("No cameras found. Make sure the camera is connected.")
        return False

    # List available cameras
    for i in range(num_cameras):
        try:
            camera = asi.Camera(i)
            camera_info = camera.get_camera_property()
            print(f"Camera {i}: {camera_info['Name']}")
            camera.close()
        except Exception as e:
            print(f"Error accessing camera {i}: {e}")

    try:
        # Create ObservatoryCamera instance for testing
        print(f"\nInitializing {camera_type} camera for testing...")

        # Create output directory in container
        output_dir = "/tmp/test_output"
        os.makedirs(output_dir, exist_ok=True)

        obs_camera = ObservatoryCamera(
            camera_id=0,
            camera_name="test-camera",
            camera_type=camera_type,
            s3_bucket=None,  # No S3 bucket for testing
            cleanup_days=7,
            skip_s3_upload=True,  # Skip S3 upload for testing
        )

        print("\nTesting image capture...")

        # Use provided settings for testing
        import astropy.units as u

        test_settings = {
            "exposure": exposure_time * u.second,
            "gain": gain,
            "mode": "test",
            "interval": 60 * u.second,
        }

        print(
            f"Capturing with {test_settings['exposure']}, gain {test_settings['gain']}"
        )

        # Capture image
        image_data = obs_camera.capture_image(
            test_settings["exposure"], test_settings["gain"]
        )

        if image_data is not None:
            print("✓ Image capture successful!")
            print(f"Image data type: {type(image_data)}")
            if hasattr(image_data, "shape"):
                print(f"Image shape: {image_data.shape}")
            else:
                print(f"Image data length: {len(image_data)}")

            # Save image (will save to /tmp/ since S3 is disabled)
            print("\nSaving image...")
            obs_camera.save_image_to_s3(image_data, test_settings)

            print("\n✓ Test completed successfully!")
            print("Images saved to /tmp/:")

            # List the created files
            try:
                files = [f for f in os.listdir("/tmp") if f.startswith("test-camera")]
                if files:
                    print("Created files:")
                    for file in sorted(files):
                        full_path = f"/tmp/{file}"
                        size_kb = os.path.getsize(full_path) / 1024
                        print(f"  - {file} ({size_kb:.1f} KB)")
                else:
                    print("  No files found with expected naming pattern")
            except Exception as e:
                print(f"  Error listing files: {e}")

            print("\n💡 Files are accessible from the host at the mounted volume")
            print("   (Check your Docker compose mount configuration)")

        else:
            print("✗ Image capture failed!")
            return False

    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        try:
            obs_camera.camera.close()
        except:
            pass

    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test ASI camera capture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_color_camera.py --exposure 0.01 --gain 200
  python test_color_camera.py --exposure 2.0 --gain 300 --camera-type color
  python test_color_camera.py --exposure 0.001 --gain 100 --camera-type mono
""",
    )

    parser.add_argument(
        "--exposure",
        "-e",
        type=float,
        default=0.01,
        help="Exposure time in seconds (default: 0.01)",
    )

    parser.add_argument(
        "--gain", "-g", type=int, default=200, help="Camera gain (default: 200)"
    )

    parser.add_argument(
        "--camera-type",
        "-t",
        choices=["color", "mono"],
        default="color",
        help="Camera type: color or mono (default: color)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=== ASI Camera Test ===")
    print(f"Testing {args.camera_type} camera with:")
    print(f"  Exposure: {args.exposure}s")
    print(f"  Gain: {args.gain}")
    print(f"  Camera type: {args.camera_type}")
    print("Images will be saved to /tmp/ inside container\n")

    success = test_camera(
        exposure_time=args.exposure, gain=args.gain, camera_type=args.camera_type
    )

    if success:
        print("\n🎉 Test passed! Your camera setup is working correctly.")
    else:
        print("\n❌ Test failed. Please check the error messages above.")
        sys.exit(1)
