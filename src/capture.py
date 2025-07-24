import datetime
import json
import os
import time

import astropy.units as u
import boto3
import numpy as np
import requests
import zwoasi as asi
from botocore.exceptions import NoCredentialsError
from dotenv import load_dotenv
from PIL import Image

from almanac import Almanac
from datetime_manager import DateTimeManager

load_dotenv()

# Initialize
asi.init("/usr/local/lib/libASICamera2.so")


class ObservatoryCamera:
    def __init__(
        self,
        camera_id=0,
        camera_name="b14m11",
        s3_bucket=None,
        cleanup_days=7,
    ):
        self.camera = asi.Camera(camera_id)
        self.camera_info = self.camera.get_camera_property()
        self.camera_name = camera_name
        self.s3_bucket = s3_bucket
        self.cleanup_days = cleanup_days

        # Initialize time management and almanac
        self.dt_manager = DateTimeManager(mode="real", timezone_str="America/Santiago")
        self.current_date = None
        self.almanac = None
        self.update_almanac()

        # Initialize S3 client
        self.s3_client = None
        if s3_bucket:
            try:
                self.s3_client = boto3.client("s3")
                print(f"S3 bucket configured: {s3_bucket}")
            except NoCredentialsError:
                print("S3 credentials not found - cannot upload images")
                return

        print(f"Using camera: {self.camera_info['Name']}")
        self.print_schedule()

    def get_observing_date(self):
        """Get the observing date (accounts for nights crossing midnight)"""
        current_time = self.dt_manager.get_current_time()

        # If it's before noon, we're still in the previous night's observing session
        if current_time.hour < 12:
            observing_date = (current_time - datetime.timedelta(days=1)).strftime(
                "%Y-%m-%d"
            )
        else:
            observing_date = current_time.strftime("%Y-%m-%d")

        return observing_date

    def update_almanac(self):
        """Update almanac if observing date has changed"""
        current_time = self.dt_manager.get_current_time()
        observing_date = self.get_observing_date()  # Use observing date instead

        # print(f"Current time: {current_time}")
        # print(f"Observing date: {observing_date}")
        # print(f"Stored date: {self.current_date}")

        if self.current_date != observing_date:
            print(
                f"Observing date changed from {self.current_date} to {observing_date}"
            )
            self.current_date = observing_date
            self.almanac = Almanac(self.dt_manager, observing_date)
            print(f"\n=== Updated almanac for observing night {observing_date} ===")
            self.print_schedule()

    def get_mount_status(self):
        """Fetch current mount status from local API"""
        try:
            response = requests.get(
                "http://localhost:5500/mount/cached-pointing", timeout=5
            )
            if response.status_code == 200:
                payload = response.json()
                return {
                    "mount_alt": payload.get("mount_alt"),
                    "mount_az": payload.get("mount_az"),
                    "mount_ra": payload.get("mount_ra"),
                    "mount_dec": payload.get("mount_dec"),
                    "connected": payload.get("connected", "UNKNOWN"),
                    "tracking": payload.get("tracking", "UNKNOWN"),
                }
            return None
        except Exception as e:
            print(f"Failed to get mount status: {e}")
            return None

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

        # The key insight: our almanac changes at noon, but we need to know if we're in
        # "today's daylight" or "tonight's darkness" or "tomorrow's daylight"

        current_hour = current_time.hour

        # Case 1: Before noon - we're using yesterday's almanac, so times are for "last night"
        # If current_time > sunrise, we're in "this morning's" daylight
        if current_hour < 12:
            is_daylight = current_time > self.almanac.sunrise

        # Case 2: After noon - we're using today's almanac, so times are for "tonight"
        # If current_time < sunset, we're still in "this afternoon's" daylight
        else:
            is_daylight = current_time < self.almanac.sunset

        # Now we have a clean is_daylight boolean that works regardless of almanac timing

        # Check enclosure status first
        enclosure_status_dict = requests.get(
            "http://0.0.0.0:8010/building/14/status"
        ).json()
        enclosure_status = enclosure_status_dict.get("status", None)

        is_enclosure_open = enclosure_status == "Open"

        if not is_enclosure_open:
            if is_daylight:
                return {
                    "exposure": 0.1 * u.second,
                    "gain": 250,
                    "interval": 1 * u.hour,
                    "mode": "day_closed",
                }
            else:
                return {
                    "exposure": 90 * u.second,
                    "gain": 420,
                    "interval": 1 * u.hour,
                    "mode": "night_closed",
                }

        # Enclosure is open - determine mode based on sun position

        if is_daylight:
            return {
                "exposure": 658 * u.microsecond,
                "gain": 255,
                "interval": 1 * u.minute,
                "mode": "day_open",
            }

        # We're in darkness - now check twilight stages

        # Morning twilight stages (only apply before noon)
        if current_hour < 12:
            if (
                self.almanac.twilight_18_deg["morning"]
                < current_time
                <= self.almanac.twilight_12_deg["morning"]
            ):
                progress = self._get_progress_between_times(
                    self.almanac.twilight_18_deg["morning"],
                    self.almanac.twilight_12_deg["morning"],
                    current_time,
                )
                exposure, gain = self._interpolate_settings(
                    start_exp=10.0,
                    end_exp=2.0,
                    start_gain=420,
                    end_gain=300,
                    progress=progress,
                )
                return {
                    "exposure": exposure * u.second,
                    "gain": int(gain),
                    "interval": 90 * u.second,
                    "mode": "morning_dark_twilight",
                }

            elif (
                self.almanac.twilight_12_deg["morning"]
                < current_time
                <= self.almanac.sunrise
            ):
                progress = self._get_progress_between_times(
                    self.almanac.twilight_12_deg["morning"],
                    self.almanac.sunrise,
                    current_time,
                )
                exposure, gain = self._interpolate_settings(
                    start_exp=2.0,
                    end_exp=0.000658,
                    start_gain=300,
                    end_gain=255,
                    progress=progress,
                )
                return {
                    "exposure": exposure * u.second,
                    "gain": int(gain),
                    "interval": 2 * u.minute,
                    "mode": "morning_bright_twilight",
                }

        # Evening twilight stages (only apply after noon)
        else:
            if (
                self.almanac.sunset
                <= current_time
                < self.almanac.twilight_12_deg["evening"]
            ):
                progress = self._get_progress_between_times(
                    self.almanac.sunset,
                    self.almanac.twilight_12_deg["evening"],
                    current_time,
                )
                exposure, gain = self._interpolate_settings(
                    start_exp=0.000658,
                    end_exp=2.0,
                    start_gain=255,
                    end_gain=300,
                    progress=progress,
                )
                return {
                    "exposure": exposure * u.second,
                    "gain": int(gain),
                    "interval": 2 * u.minute,
                    "mode": "evening_bright_twilight",
                }

            elif (
                self.almanac.twilight_12_deg["evening"]
                <= current_time
                < self.almanac.twilight_18_deg["evening"]
            ):
                progress = self._get_progress_between_times(
                    self.almanac.twilight_12_deg["evening"],
                    self.almanac.twilight_18_deg["evening"],
                    current_time,
                )
                exposure, gain = self._interpolate_settings(
                    start_exp=2.0,
                    end_exp=10.0,
                    start_gain=300,
                    end_gain=420,
                    progress=progress,
                )
                return {
                    "exposure": exposure * u.second,
                    "gain": int(gain),
                    "interval": 90 * u.second,
                    "mode": "evening_dark_twilight",
                }

        # Full darkness (not in any twilight period)
        moon_is_up, moon_phase = self.almanac.is_moon_up()
        if not moon_is_up:
            return {
                "exposure": 10 * u.second,
                "gain": 420,
                "interval": 55 * u.second,
                "mode": "dark",
            }
        else:
            # Moon is up - scale exposure based on phase
            # Use aggressive scaling: moon brightness increases faster than linear

            # Scale from 10s (phase=0) to 1s (phase=1) with power function
            base_exposure = 10.0
            min_exposure = 1
            exposure_range = base_exposure - min_exposure

            # More aggressive scaling (2.5 or 3.0 exponent)
            darkness_factor = (1.0 - moon_phase) ** 3
            exposure_seconds = base_exposure - (exposure_range * darkness_factor)

            # Ensure we stay within bounds
            exposure_seconds = (
                min_exposure + (base_exposure - min_exposure) * darkness_factor
            )

            return {
                "exposure": exposure_seconds * u.second,
                "gain": 420,
                "interval": 55 * u.second,
                "mode": f"dark_moon_{moon_phase * 100:.0f}pct",
            }

    def _get_progress_between_times(self, start_time, end_time, current_time):
        """Calculate progress (0.0 to 1.0) between two times"""
        total_duration = (end_time - start_time).total_seconds()
        elapsed = (current_time - start_time).total_seconds()
        progress = max(0.0, min(1.0, elapsed / total_duration))
        return progress

    def _interpolate_settings(self, start_exp, end_exp, start_gain, end_gain, progress):
        """Linear interpolation between camera settings"""
        # Exponential interpolation for exposure (feels more natural)
        import math

        log_start = math.log(start_exp)
        log_end = math.log(end_exp)
        exposure = math.exp(log_start + progress * (log_end - log_start))

        # Linear interpolation for gain
        gain = start_gain + progress * (end_gain - start_gain)

        return exposure, gain

    def capture_image(self, exposure, gain):
        """Capture a single image with specified settings"""
        try:
            exposure_us = int(exposure.to(u.microsecond).value)
            print(f"Setting exposure to {exposure_us} microseconds, gain {gain}")

            self.camera.set_control_value(asi.ASI_EXPOSURE, exposure_us)
            self.camera.set_control_value(asi.ASI_GAIN, gain)
            self.camera.set_image_type(asi.ASI_IMG_RAW8)

            print("Starting exposure...")
            self.camera.start_exposure()

            exposure_seconds = exposure.to(u.second).value
            if exposure_seconds < 0.01:
                wait_time = 0.5
            else:
                wait_time = exposure_seconds + 0.2

            print(f"Waiting {wait_time} seconds...")
            time.sleep(wait_time)

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
                    time.sleep(0.1)
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

    def save_image_to_s3(self, image_data, settings):
        """Save image directly to S3 in all formats"""
        try:
            # Process image data
            if isinstance(image_data, bytearray):
                image_array = np.frombuffer(image_data, dtype=np.uint8)
                camera_info = self.camera.get_camera_property()
                height = camera_info["MaxHeight"]
                width = camera_info["MaxWidth"]

                try:
                    current_roi = self.camera.get_roi()
                    if current_roi:
                        width, height = current_roi[2], current_roi[3]
                except:
                    pass

                print(f"Reshaping {len(image_array)} bytes to {width}x{height}")

                try:
                    normalized = image_array.reshape((height, width))
                except ValueError as e:
                    print(f"Reshape failed: {e}")
                    total_pixels = len(image_array)
                    estimated_width = int(np.sqrt(total_pixels))
                    estimated_height = total_pixels // estimated_width
                    print(
                        f"Trying estimated dimensions: {estimated_width}x{estimated_height}"
                    )
                    normalized = image_array.reshape(
                        (estimated_height, estimated_width)
                    )

            elif isinstance(image_data, np.ndarray):
                if image_data.dtype != np.uint8:
                    normalized = (
                        (image_data - image_data.min())
                        / (image_data.max() - image_data.min())
                        * 255
                    ).astype(np.uint8)
                else:
                    normalized = image_data
            else:
                print(f"Unexpected data type: {type(image_data)}")
                return

            # Create PIL images - full resolution and thumbnailed
            img_full = Image.fromarray(normalized)
            img_thumb = img_full.copy()

            # Thumbnail only the WebP and JPEG versions
            if normalized.shape[0] > 1500 or normalized.shape[1] > 2000:
                img_thumb.thumbnail((2000, 1500), Image.Resampling.LANCZOS)
                print(
                    f"Thumbnailed from {img_full.size} to {img_thumb.size} for WebP/JPEG"
                )

            # Create safe timestamp for filename
            timestamp = self.dt_manager.get_current_time()
            safe_timestamp = timestamp.strftime("%Y%m%d_%H%M%S")

            # Create temporary files
            temp_webp = f"/tmp/{self.camera_name}-{safe_timestamp}.webp"
            temp_png = f"/tmp/{self.camera_name}-{safe_timestamp}.png"

            # Save in all formats - PNG at full resolution, others thumbnailed
            img_thumb.save(temp_webp, format="WebP", quality=75, method=6)
            img_full.save(temp_png, format="PNG")  # Full resolution PNG

            print(f"Saved WebP/JPEG at {img_thumb.size}, PNG at {img_full.size}")

            # Upload timestamped versions to S3
            self.upload_file_to_s3(
                temp_webp,
                f"{self.camera_name}/{self.camera_name}-{safe_timestamp}.webp",
                mode=settings["mode"],
            )
            self.upload_file_to_s3(
                temp_png,
                f"{self.camera_name}/{self.camera_name}-{safe_timestamp}.png",
                mode=settings["mode"],
            )

            # Copy to latest versions using S3
            self.copy_to_latest(
                f"{self.camera_name}/{self.camera_name}-{safe_timestamp}.webp",
                f"{self.camera_name}/latest.webp",
            )
            self.copy_to_latest(
                f"{self.camera_name}/{self.camera_name}-{safe_timestamp}.png",
                f"{self.camera_name}/latest.png",
            )

            # Upload status file
            self.save_status_to_s3(settings, timestamp)

            # Clean up temp files
            os.remove(temp_webp)
            os.remove(temp_png)

            exposure_display = self._format_exposure_for_display(settings["exposure"])
            print(
                f"Uploaded to S3: {self.camera_name}-{safe_timestamp} ({settings['mode']}: {exposure_display}, gain {settings['gain']})"
            )

        except Exception as e:
            print(f"Error saving image to S3: {e}")
            import traceback

            traceback.print_exc()

    def upload_file_to_s3(self, local_path, s3_key, mode):
        """Upload a file to S3"""
        try:
            # Determine content type based on extension
            if s3_key.endswith(".webp"):
                content_type = "image/webp"
            elif s3_key.endswith(".jpg"):
                content_type = "image/jpeg"
            elif s3_key.endswith(".png"):
                content_type = "image/png"
            elif s3_key.endswith(".json"):
                content_type = "application/json"
            else:
                content_type = "binary/octet-stream"

            self.s3_client.upload_file(
                local_path,
                self.s3_bucket,
                s3_key,
                ExtraArgs={"ContentType": content_type, "Tagging": f"mode={mode}"},
            )
            print(f"Uploaded: s3://{self.s3_bucket}/{s3_key}")

        except Exception as e:
            print(f"Failed to upload {s3_key}: {e}")

    def copy_to_latest(self, source_key, dest_key):
        """Copy a file to the 'latest' version using S3 copy"""
        try:
            copy_source = {"Bucket": self.s3_bucket, "Key": source_key}

            # Determine content type
            if dest_key.endswith(".webp"):
                content_type = "image/webp"
            elif dest_key.endswith(".jpg"):
                content_type = "image/jpeg"
            elif dest_key.endswith(".png"):
                content_type = "image/png"
            else:
                content_type = "binary/octet-stream"

            self.s3_client.copy_object(
                CopySource=copy_source,
                Bucket=self.s3_bucket,
                Key=dest_key,
                MetadataDirective="REPLACE",
                ContentType=content_type,
            )
            print(f"Copied to latest: {dest_key}")

        except Exception as e:
            print(f"Failed to copy to {dest_key}: {e}")

    def save_status_to_s3(self, settings, timestamp):
        """Save status information directly to S3"""
        try:
            # Get mount status
            mount_status = self.get_mount_status()

            # Create status description based on mode
            if settings["mode"] == "day_closed":
                description = (
                    "Observatory closed during daytime. Images update every hour."
                )
            elif settings["mode"] in [
                "evening_bright_twilight",
                "morning_bright_twilight",
            ]:
                description = "Twilight period. Images update every 2 minutes."
            elif settings["mode"] in ["evening_dark_twilight", "morning_dark_twilight"]:
                description = "Deep twilight. Images update every 90 seconds."
            elif settings["mode"] == "dark":
                description = "Dark sky observing. Images update every 60 seconds."
            else:
                description = f"Observatory operating in {settings['mode']} mode."

            status_data = {
                "camera_name": self.camera_name,
                "last_update": timestamp.isoformat(),
                "last_update_friendly": timestamp.strftime("%Y-%m-%d %H:%M:%S %Z"),
                "mode": settings["mode"],
                "description": description,
                "exposure_time": str(settings["exposure"]),
                "gain": settings["gain"],
                "next_update_interval": str(settings["interval"]),
                "status": "online",
                "mount_status": mount_status,  # Add mount status
            }

            # Save to temp file and upload
            temp_status = f"/tmp/{self.camera_name}_status.json"
            with open(temp_status, "w") as f:
                json.dump(status_data, f, indent=2)

            self.upload_file_to_s3(
                temp_status,
                f"{self.camera_name}/{self.camera_name}_status.json",
                mode=status_data["mode"],
            )
            os.remove(temp_status)

        except Exception as e:
            print(f"Error saving status to S3: {e}")

    def _format_exposure_for_display(self, exposure):
        """Format exposure time for human-readable display"""
        if exposure >= 1 * u.second:
            return f"{exposure.to(u.second):.1f}"
        elif exposure >= 1 * u.millisecond:
            return f"{exposure.to(u.millisecond):.1f}"
        else:
            return f"{exposure.to(u.microsecond):.0f}"

    def cleanup_old_files(self):
        """Clean up old timestamped files from S3"""
        try:
            cutoff_time = self.dt_manager.get_current_time() - datetime.timedelta(
                days=self.cleanup_days
            )
            cutoff_str = cutoff_time.strftime("%Y%m%d_%H%M%S")

            # List objects with the camera name prefix
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket, Prefix=f"{self.camera_name}-"
            )

            if "Contents" not in response:
                return

            deleted_count = 0
            for obj in response["Contents"]:
                key = obj["Key"]
                # Extract timestamp from filename like "b14m11-20250702_143022.webp"
                if "-" in key and "." in key:
                    try:
                        timestamp_part = key.split("-")[1].split(".")[
                            0
                        ]  # Get "20250702_143022"
                        if timestamp_part < cutoff_str:
                            self.s3_client.delete_object(Bucket=self.s3_bucket, Key=key)
                            deleted_count += 1
                            print(f"Deleted old file: {key}")
                    except:
                        pass  # Skip files that don't match expected format

            if deleted_count > 0:
                print(f"Cleaned up {deleted_count} old files from S3")

        except Exception as e:
            print(f"Error during S3 cleanup: {e}")

    def run_continuous(self):
        """Main capture loop with responsive time checking"""
        last_cleanup = self.dt_manager.get_current_time() - datetime.timedelta(days=1)
        last_capture_time = None

        try:
            while True:
                self.update_almanac()

                current_time = self.dt_manager.get_current_time()

                # Daily cleanup
                if (current_time - last_cleanup).days >= 1:
                    self.cleanup_old_files()
                    last_cleanup = current_time

                # Get current settings
                settings = self.get_camera_settings()

                # Determine if we should capture now
                should_capture = False

                if last_capture_time is None:
                    # First capture
                    should_capture = True
                    reason = "Initial capture"
                else:
                    # Check if enough time has passed since last capture
                    time_since_capture = (
                        current_time - last_capture_time
                    ).total_seconds()
                    required_interval = settings["interval"].to(u.second).value

                    if time_since_capture >= required_interval:
                        should_capture = True
                        reason = f"Interval reached ({time_since_capture:.0f}s >= {required_interval}s)"
                    else:
                        # Check if we've switched to a different mode (which might need different cadence)
                        previous_settings = getattr(self, "_last_settings", None)
                        if (
                            previous_settings
                            and previous_settings["mode"] != settings["mode"]
                        ):
                            should_capture = True
                            reason = f"Mode changed: {previous_settings['mode']} → {settings['mode']}"

                if should_capture:
                    print(
                        f"\n{current_time.strftime('%Y-%m-%d %H:%M:%S')} - Mode: {settings['mode']}"
                    )
                    print(f"Capture reason: {reason}")

                    image_data = self.capture_image(
                        settings["exposure"], settings["gain"]
                    )

                    if image_data is not None:
                        self.save_image_to_s3(image_data, settings)
                        last_capture_time = current_time
                        self._last_settings = settings.copy()
                    else:
                        print("Failed to capture image")
                else:
                    # Show status occasionally without spamming
                    if int(time.time()) % 300 == 0:  # Every 30 seconds
                        time_until_next = (
                            settings["interval"].to(u.second).value
                            - (current_time - last_capture_time).total_seconds()
                        )
                        print(
                            f"{current_time.strftime('%H:%M:%S')} - Mode: {settings['mode']}, next capture in {time_until_next:.0f}s"
                        )

                # Always sleep for just 1 second to check again
                time.sleep(1)

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

    obs_camera = ObservatoryCamera(
        camera_id=0,
        camera_name="b14m11",
        s3_bucket=os.environ.get("S3_BUCKET_NAME"),
        cleanup_days=7,
    )

    obs_camera.run_continuous()


if __name__ == "__main__":
    main()
