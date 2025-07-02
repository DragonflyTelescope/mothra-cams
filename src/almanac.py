import datetime
import subprocess as sp
from datetime import timedelta

import astropy.units as u
import ephem
import numpy as np
import pytz
from astropy.coordinates import EarthLocation


# from .site import get_nms_sitelocation
def get_nms_sitelocation():
    info = {
        "elevation": 2225.04,
        "latitude": 32.902040,
        "longitude": -105.530640,
        "timezone": "America/Denver",
    }
    return EarthLocation(
        lat=info["latitude"] * u.deg,
        lon=info["longitude"] * u.deg,
        height=info["elevation"] * u.m,
    )


def get_nms_temperature():
    r = sp.run(
        "curl -s https://nmskies.com/weather1.txt", shell=True, capture_output=True
    )
    use_str = r.stdout.decode("utf-8")
    words = use_str.split(",")
    temperature = words[0]
    if len(temperature) == 0:
        return None
    return int(temperature)


def barometric_pressure(h: u.Quantity):
    p0 = 101325 * u.Pa
    L = 0.00976 * (u.K / u.m)
    T0 = 288.16 * u.K
    g = 9.80665 * (u.m / u.s**2)
    M = 0.02896968 * (u.kg / u.mol)
    R0 = 8.314462618 * (u.J / (u.mol * u.K))
    pressure = p0 * (1 - (L * h / T0)) ** (g * M / (R0 * L))
    return pressure.to(u.bar)


class Almanac:
    def __init__(self, dt_manager, date_str, morning_bailout_buffer=1):
        self.dt_manager = dt_manager
        self.site = ephem.Observer()
        self.site.lat, self.site.lon = (
            "32.902040",
            "-105.530640",
        )  # Adjust to your observatory coordinates
        self.site.elevation = 2225.04
        self.site.pressure = 0
        self.site.horizon = "-0:34"

        # Calculate the correct midnight (the start of the next day from the given date)
        self.date = date_str
        self.morning_bailout = morning_bailout_buffer
        observing_date = datetime.datetime.strptime(
            date_str, "%Y-%m-%d"
        ) + datetime.timedelta(days=1)
        local_midnight = observing_date.replace(
            hour=0, minute=0, second=0, microsecond=0, tzinfo=self.dt_manager.timezone
        )
        utc_midnight = local_midnight.astimezone(pytz.utc)

        self.site.date = utc_midnight.strftime("%Y/%m/%d %H:%M:%S")

        self.compute_times()

    def compute_times(self):
        sun = ephem.Sun()
        moon = ephem.Moon()

        self.site.horizon = "-0:34"  # Normal horizon level for sunsets and sunrises
        sunset = (
            self.site.previous_setting(sun, use_center=True)
            .datetime()
            .replace(tzinfo=pytz.utc)
            .astimezone(self.dt_manager.timezone)
        )
        sunrise = (
            self.site.next_rising(sun, use_center=True)
            .datetime()
            .replace(tzinfo=pytz.utc)
            .astimezone(self.dt_manager.timezone)
        )
        self.sunset = sunset
        self.sunrise = sunrise

        # Compute evening and morning twilight times
        self.site.horizon = "-6"
        evening_twilight_6_deg = (
            self.site.previous_setting(sun, use_center=True)
            .datetime()
            .replace(tzinfo=pytz.utc)
            .astimezone(self.dt_manager.timezone)
        )
        morning_twilight_6_deg = (
            self.site.next_rising(sun, use_center=True)
            .datetime()
            .replace(tzinfo=pytz.utc)
            .astimezone(self.dt_manager.timezone)
        )

        self.site.horizon = "-12"
        evening_twilight_12_deg = (
            self.site.previous_setting(sun, use_center=True)
            .datetime()
            .replace(tzinfo=pytz.utc)
            .astimezone(self.dt_manager.timezone)
        )
        morning_twilight_12_deg = (
            self.site.next_rising(sun, use_center=True)
            .datetime()
            .replace(tzinfo=pytz.utc)
            .astimezone(self.dt_manager.timezone)
        )

        self.site.horizon = "-18"
        evening_twilight_18_deg = (
            self.site.previous_setting(sun, use_center=True)
            .datetime()
            .replace(tzinfo=pytz.utc)
            .astimezone(self.dt_manager.timezone)
        )
        morning_twilight_18_deg = (
            self.site.next_rising(sun, use_center=True)
            .datetime()
            .replace(tzinfo=pytz.utc)
            .astimezone(self.dt_manager.timezone)
        )

        # Assign the computed twilight times
        self.twilight_6_deg = {
            "evening": evening_twilight_6_deg,
            "morning": morning_twilight_6_deg,
        }
        self.twilight_12_deg = {
            "evening": evening_twilight_12_deg,
            "morning": morning_twilight_12_deg,
        }
        self.twilight_18_deg = {
            "evening": evening_twilight_18_deg,
            "morning": morning_twilight_18_deg,
        }
        # Reset horizon for accurate moonrise/set calculations
        self.site.horizon = "-0:34"
        moonrise = (
            self.site.next_rising(moon, use_center=True)
            .datetime()
            .replace(tzinfo=pytz.utc)
            .astimezone(self.dt_manager.timezone)
        )
        moonset = (
            self.site.next_setting(moon, use_center=True)
            .datetime()
            .replace(tzinfo=pytz.utc)
            .astimezone(self.dt_manager.timezone)
        )

        # Check if moonrise and moonset are between sunset and sunrise
        if sunset <= moonrise <= sunrise:
            self.moonrise = moonrise
        else:
            self.moonrise = None

        if sunset <= moonset <= sunrise:
            self.moonset = moonset
        else:
            self.moonset = None

    @property
    def current_time(self):
        return self.dt_manager.get_current_time()

    @property
    def is_morning(self):
        return self.current_time > self.twilight_12_deg["morning"]

    @property
    def bailout_time(self):
        return self.twilight_12_deg["morning"] - timedelta(hours=self.morning_bailout)

    @property
    def time_to_bailout(self):
        return self.bailout_time - self.current_time

    @property
    def is_twilight_0_12(self):
        return (self.current_time > self.sunset) and (
            self.current_time < self.twilight_12_deg["evening"]
        )

    @property
    def is_twilight_12_18(self):
        return (self.current_time > self.twilight_12_deg["evening"]) and (
            self.current_time < self.twilight_18_deg["evening"]
        )

    @property
    def max_flat_time(self):
        return self.sunset + timedelta(minutes=50)

    @property
    def min_flat_time(self):
        return self.sunset + timedelta(minutes=6)

    @property
    def is_pre_flat_window(self):
        return self.current_time < self.min_flat_time

    @property
    def is_post_flat_window(self):
        return self.current_time > self.max_flat_time

    @property
    def is_past_12_deg(self):
        return self.current_time > self.twilight_12_deg["evening"]

    @property
    def is_past_18_deg(self):
        return self.current_time > self.twilight_18_deg["evening"]

    def _create_fresh_observer(self, datetime_arg):
        """
        Create a fresh ephem.Observer with the site parameters but updated date.

        Args:
            datetime_arg (str, datetime or None): Time specification to use.

        Returns:
            tuple: (ephem.Observer, datetime) - Observer with updated time and the calculation time
        """
        # Create fresh observer
        fresh_site = ephem.Observer()
        fresh_site.lat = self.site.lat
        fresh_site.lon = self.site.lon
        fresh_site.elevation = self.site.elevation
        fresh_site.pressure = self.site.pressure
        fresh_site.horizon = self.site.horizon

        # Determine the time to use for calculation
        if datetime_arg == "now" or datetime_arg is None:
            # Use current time
            calculation_time = self.dt_manager.get_current_time()
        elif datetime_arg == "sunset":
            # Use sunset time for today
            calculation_time = self.sunset
        else:
            # Parse the provided datetime string
            try:
                # Parse the local time string
                naive_dt = datetime.datetime.strptime(
                    datetime_arg, "%Y-%m-%d %H:%M:%S.%f"
                )
                # Attach the timezone
                calculation_time = naive_dt.replace(tzinfo=self.dt_manager.timezone)
            except ValueError:
                # Try without microseconds
                try:
                    naive_dt = datetime.datetime.strptime(
                        datetime_arg, "%Y-%m-%d %H:%M:%S"
                    )
                    calculation_time = naive_dt.replace(tzinfo=self.dt_manager.timezone)
                except ValueError:
                    raise ValueError(
                        "Invalid datetime format. Expected 'YYYY-MM-DD HH:MM:SS.S' or 'YYYY-MM-DD HH:MM:SS'"
                    )

        # Set the site's date to the calculation time in UTC
        fresh_site.date = calculation_time.astimezone(pytz.utc).strftime(
            "%Y/%m/%d %H:%M:%S"
        )

        return fresh_site, calculation_time

    def solar_position(self, datetime_arg=None):
        """
        Compute the solar position (both altitude and azimuth).

        Args:
            datetime_arg (str or None): Either a datetime string in format "YYYY-MM-DD HH:MM:SS.S",
                                    "sunset" to use sunset time, or None/"now" to use current time.

        Returns:
            tuple: (solar_altitude, solar_azimuth) in degrees.
        """
        # Create fresh observer with the appropriate time
        fresh_site, _ = self._create_fresh_observer(datetime_arg)

        # Compute the Sun's position
        sun = ephem.Sun()
        sun.compute(fresh_site)

        # Get the Sun's altitude and azimuth
        sun_altitude = np.rad2deg(sun.alt)  # Convert from radians to degrees
        sun_azimuth = np.rad2deg(sun.az)  # Convert from radians to degrees

        return (sun_altitude, sun_azimuth)

    def antisolar_position(self, datetime_arg=None, arago_offset=20):
        """
        Compute the antisolar point and Arago point (null point) positions.

        Args:
            datetime_arg (str or None): Either a datetime string in format "YYYY-MM-DD HH:MM:SS.S",
                                    "sunset" to use sunset time, or None/"now" to use current time.
            arago_offset (float): Angular offset of Arago point above antisolar point in degrees.
                                Default is 20 degrees.

        Returns:
            dict: Dictionary containing:
                - antisolar_alt: Altitude of antisolar point in degrees
                - antisolar_az: Azimuth of antisolar point in degrees
                - arago_alt: Altitude of Arago point in degrees
                - arago_az: Azimuth of Arago point in degrees (same as antisolar_az)
                - time: The datetime used for calculation
        """
        # Create fresh observer with the appropriate time
        fresh_site, calculation_time = self._create_fresh_observer(datetime_arg)

        # Compute the Sun's position
        sun = ephem.Sun()
        sun.compute(fresh_site)

        # Get the Sun's altitude and azimuth
        sun_altitude = np.rad2deg(sun.alt)  # Convert from radians to degrees
        sun_azimuth = np.rad2deg(sun.az)  # Convert from radians to degrees

        # Calculate antisolar point (opposite azimuth, negative altitude)
        antisolar_azimuth = (sun_azimuth + 180) % 360
        antisolar_altitude = -sun_altitude

        # Calculate Arago point (typically ~20° above the antisolar point)
        arago_azimuth = antisolar_azimuth
        arago_altitude = antisolar_altitude + arago_offset

        return {
            "antisolar_alt": antisolar_altitude,
            "antisolar_az": antisolar_azimuth,
            "arago_alt": arago_altitude,
            "arago_az": arago_azimuth,
            "time": calculation_time,
        }

    def arago_point(self, datetime_arg=None, arago_offset=20):
        """
        Compute the Arago point (null point) position.
        Convenience method that returns just the Arago point coordinates.

        Args:
            datetime_arg (str or None): Either a datetime string in format "YYYY-MM-DD HH:MM:SS.S",
                                    "sunset" to use sunset time, or None/"now" to use current time.
            arago_offset (float): Angular offset of Arago point above antisolar point in degrees.
                                Default is 20 degrees.

        Returns:
            tuple: (arago_altitude, arago_azimuth) in degrees
        """
        result = self.antisolar_position(datetime_arg, arago_offset)
        return (result["arago_alt"], result["arago_az"])
