import datetime
import os
import time

import astropy.units as u
import boto3
import numpy as np
import zwoasi as asi
from botocore.exceptions import NoCredentialsError
from PIL import Image

from almanac import Almanac

# Import your utilities
from datetime_manager import DateTimeManager

# Initialize
asi.init("/usr/local/lib/libASICamera2.so")


class ObservatoryCamera:
    def __init__(
        self,
        camera_id=0,
        output_dir="/data/observatory/img",
        camera_name="b14m11",
        s3_bucket=None,
        cleanup_days=7,
    ):
        self.camera = asi.Camera(camera_id)
        self.camera_info = self.camera.get_camera_property()
        self.output_dir = output_dir
        self.camera_name = camera_name
        self.s3_bucket = s3_bucket
        self.cleanup_days = cleanup_days

        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "archive"), exist_ok=True)

        # Initialize time management and almanac
        self.dt_manager = DateTimeManager(mode="real", timezone_str="America/Santiago")
        self.current_date = None
        self.almanac = None
        self.update_almanac()

        # Initialize S3 client if bucket specified
        self.s3_client = None
        if s3_bucket:
            try:
                self.s3_client = boto3.client("s3")
                print(f"S3 bucket configured: {s3_bucket}")
            except NoCredentialsError:
                print("S3 credentials not found - timelapse archiving disabled")

        print(f"Using camera: {self.camera_info['Name']}")
        self.print_schedule()

    def update_almanac(self):
        """Update almanac if date has changed"""
        current_time = self.dt_manager.get_current_time()
        today_str = current_time.strftime("%Y-%m-%d")

        if self.current_date != today_str:
            self.current_date = today_str
            self.almanac = Almanac(self.dt_manager, today_str)
            print(f"\n=== Updated almanac for {today_str} ===")
            self.print_schedule()

    def print_schedule(self):
        """Print today's astronomical schedule"""
        print(f"Sunset: {self.almanac.sunset.strftime('%H:%M')}")
        print(
            f"12° twilight: {self.almanac.twilight_12_deg['evening'].strftime('%H:%M')}"
        )
        print(
            f"18° twilight: {self.almanac.twilight_18_deg['evening'].strftime('%H:%M')}"
        )
        print(
            f"18° morning twilight: {self.almanac.twilight_18_deg['morning'].strftime('%H:%M')}"
        )
        print(
            f"12° morning twilight: {self.almanac.twilight_12_deg['morning'].strftime('%H:%M')}"
        )
        print(f"Sunrise: {self.almanac.sunrise.strftime('%H:%M')}")

    def get_camera_settings(self):
        """Determine camera settings based on current time"""
        current_time = self.dt_manager.get_current_time()

        # Handle morning twilight back to day
        if current_time > self.almanac.sunrise:
            # Full daytime - roof closed, minimal settings
            return {
                "exposure": 658 * u.microsecond,
                "gain": 255,
                "interval": 1 * u.hour,
                "mode": "day_closed",
                "save_timelapse": False,
            }

        elif current_time > self.almanac.twilight_12_deg["morning"]:
            # Morning 12° to sunrise - bright morning twilight
            return {
                "exposure": 1 * u.second,
                "gain": 100,
                "interval": 2 * u.minute,
                "mode": "morning_bright_twilight",
                "save_timelapse": False,
            }

        elif current_time > self.almanac.twilight_18_deg["morning"]:
            # Morning 18° to 12° twilight
            return {
                "exposure": 5 * u.second,
                "gain": 250,
                "interval": 90 * u.second,
                "mode": "morning_dark_twilight",
                "save_timelapse": True,
            }

        # Evening schedule
        elif current_time < self.almanac.sunset:
            # Before sunset - daytime (roof open)
            return {
                "exposure": 658 * u.microsecond,
                "gain": 255,
                "interval": 1 * u.hour,
                "mode": "day_open",
                "save_timelapse": False,
            }

        elif current_time < self.almanac.twilight_12_deg["evening"]:
            # Sunset to 12° twilight
            return {
                "exposure": 1 * u.second,
                "gain": 100,
                "interval": 2 * u.minute,
                "mode": "evening_bright_twilight",
                "save_timelapse": False,
            }

        elif current_time < self.almanac.twilight_18_deg["evening"]:
            # 12° to 18° twilight
            return {
                "exposure": 5 * u.second,
                "gain": 250,
                "interval": 90 * u.second,
                "mode": "evening_dark_twilight",
                "save_timelapse": True,
            }

        else:
            # Past 18° twilight - full dark
            return {
                "exposure": 10 * u.second,
                "gain": 420,
                "interval": 55 * u.second,
                "mode": "dark",
                "save_timelapse": True,
            }

    def capture_image(self, exposure, gain):
        """Capture a single image with specified settings"""
        try:
            exposure_us = int(exposure.to(u.microsecond).value)
            print(f"Setting exposure to {exposure_us} microseconds, gain {gain}")

            # Set camera settings
            self.camera.set_control_value(asi.ASI_EXPOSURE, exposure_us)
            self.camera.set_control_value(asi.ASI_GAIN, gain)
            self.camera.set_image_type(asi.ASI_IMG_RAW8)

            print("Starting exposure...")
            self.camera.start_exposure()

            # For very short exposures, we need more buffer time for camera processing
            exposure_seconds = exposure.to(u.second).value
            if exposure_seconds < 0.01:  # Less than 10ms
                wait_time = 0.5  # Give it 500ms for very short exposures
            else:
                wait_time = exposure_seconds + 0.1

            print(f"Waiting {wait_time} seconds...")
            time.sleep(wait_time)

            # Poll the status until it's ready
            max_attempts = 20
            for attempt in range(max_attempts):
                status = self.camera.get_exposure_status()
                print(f"Attempt {attempt + 1}: Exposure status: {status}")

                if status == asi.ASI_EXP_SUCCESS:
                    print("Exposure successful!")
                    return self.camera.get_data_after_exposure()
                elif status == asi.ASI_EXP_FAILED:
                    print("Exposure failed")
                    return None
                elif status == asi.ASI_EXP_WORKING:
                    print("Still working... waiting more")
                    time.sleep(0.1)  # Wait another 100ms
                else:
                    print(f"Unknown status: {status}")
                    time.sleep(0.1)

            print("Timed out waiting for exposure to complete")
            return None

        except Exception as e:
            print(f"Error during capture: {e}")
            import traceback

            traceback.print_exc()
            return None

    def save_image(self, image_data, settings):
        """Save image data as PNG"""
        try:
            # Normalize image
            if image_data.dtype != np.uint8:
                normalized = (
                    (image_data - image_data.min())
                    / (image_data.max() - image_data.min())
                    * 255
                ).astype(np.uint8)
            else:
                normalized = image_data

            img = Image.fromarray(normalized, mode="L")

            # Always save/overwrite the main image for the website
            main_filename = os.path.join(self.output_dir, f"{self.camera_name}.png")
            img.save(main_filename)

            # Save timestamped archive for timelapse (only during night)
            if settings["save_timelapse"]:
                timestamp = self.dt_manager.get_current_time()
                archive_filename = os.path.join(
                    self.output_dir,
                    "archive",
                    f"{self.camera_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}_{settings['mode']}.png",
                )
                img.save(archive_filename)

                # Also upload to S3 for timelapse storage
                if self.s3_client and self.s3_bucket:
                    self.upload_to_s3(archive_filename, timestamp)

            # Format exposure for display
            exposure_display = self._format_exposure_for_display(settings["exposure"])
            print(
                f"Saved {main_filename} ({settings['mode']}: {exposure_display}, gain {settings['gain']})"
            )

        except Exception as e:
            print(f"Error saving image: {e}")

    def _format_exposure_for_display(self, exposure):
        """Format exposure time for human-readable display"""
        if exposure >= 1 * u.second:
            return f"{exposure.to(u.second):.1f}"
        elif exposure >= 1 * u.millisecond:
            return f"{exposure.to(u.millisecond):.1f}"
        else:
            return f"{exposure.to(u.microsecond):.0f}"

    def upload_to_s3(self, local_filename, timestamp):
        """Upload image to S3 for timelapse archive"""
        try:
            date_str = timestamp.strftime("%Y-%m-%d")
            s3_key = (
                f"observatory-timelapse/{date_str}/{os.path.basename(local_filename)}"
            )

            self.s3_client.upload_file(local_filename, self.s3_bucket, s3_key)
            print(f"Uploaded to S3: s3://{self.s3_bucket}/{s3_key}")

        except Exception as e:
            print(f"Failed to upload to S3: {e}")

    def cleanup_old_files(self):
        """Clean up old archive files"""
        try:
            cutoff_date = self.dt_manager.get_current_time() - datetime.timedelta(
                days=self.cleanup_days
            )
            archive_dir = os.path.join(self.output_dir, "archive")

            deleted_count = 0
            for filename in os.listdir(archive_dir):
                if filename.endswith(".png"):
                    file_path = os.path.join(archive_dir, filename)
                    file_time = datetime.datetime.fromtimestamp(
                        os.path.getctime(file_path), tz=self.dt_manager.timezone
                    )

                    if file_time < cutoff_date:
                        os.remove(file_path)
                        deleted_count += 1

            if deleted_count > 0:
                print(f"Cleaned up {deleted_count} old archive files")

        except Exception as e:
            print(f"Error during cleanup: {e}")

    def deploy_to_github(self):
        """Deploy the latest image to GitHub Pages"""
        try:
            import os
            import subprocess

            repo_dir = "/repo"
            token = os.getenv("GITHUB_TOKEN")
            repo_url = os.getenv("GITHUB_REPO_URL")

            if not token or not repo_url:
                print("❌ GitHub token or repo URL not configured")
                return

            # Configure git
            subprocess.run(
                [
                    "git",
                    "config",
                    "user.name",
                    os.getenv("GIT_AUTHOR_NAME", "Observatory"),
                ],
                cwd=repo_dir,
            )
            subprocess.run(
                [
                    "git",
                    "config",
                    "user.email",
                    os.getenv("GIT_AUTHOR_EMAIL", "obs@example.com"),
                ],
                cwd=repo_dir,
            )

            # Set remote URL with token authentication
            auth_url = repo_url.replace("https://", f"https://{token}@")
            subprocess.run(
                ["git", "remote", "set-url", "origin", auth_url], cwd=repo_dir
            )

            # Deploy with ghp-import
            result = subprocess.run(
                ["ghp-import", "-p", "html/"],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=60,  # 60 second timeout
            )

            if result.returncode == 0:
                print("✅ Successfully deployed to GitHub Pages")
            else:
                print(f"❌ Deploy failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            print("❌ Deploy timed out")
        except Exception as e:
            print(f"❌ Error deploying: {e}")

    def run_continuous(self):
        """Main capture loop"""
        # Fix: Use timezone-aware datetime instead of datetime.min
        last_cleanup = self.dt_manager.get_current_time() - datetime.timedelta(days=1)

        try:
            while True:
                # Update almanac if date changed
                self.update_almanac()

                # Cleanup old files once per day
                current_time = self.dt_manager.get_current_time()
                if (current_time - last_cleanup).days >= 1:
                    self.cleanup_old_files()
                    last_cleanup = current_time

                # Get current settings
                settings = self.get_camera_settings()

                print(
                    f"\n{current_time.strftime('%Y-%m-%d %H:%M:%S')} - Mode: {settings['mode']}"
                )

                # Capture and save image
                image_data = self.capture_image(settings["exposure"], settings["gain"])

                if image_data is not None:
                    self.save_image(image_data, settings)
                else:
                    print("Failed to capture image")

                # Wait until next capture (convert interval to seconds)
                interval_seconds = settings["interval"].to(u.second).value
                print(f"Waiting {settings['interval']} until next capture...")
                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print("\nStopping capture...")
        finally:
            self.camera.close()
            print("Camera closed")


def main():
    num_cameras = asi.get_num_cameras()
    if num_cameras == 0:
        print("No cameras found")
        return

    # Configuration
    obs_camera = ObservatoryCamera(
        camera_id=0,
        output_dir="/data/observatory/img",
        camera_name="b14m11",
        s3_bucket=None,  # Set to None to disable S3
        cleanup_days=1,
    )

    obs_camera.run_continuous()


if __name__ == "__main__":
    main()
