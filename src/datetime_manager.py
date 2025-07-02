import datetime
import sys
import time

import pytz


class DateTimeManager:
    def __init__(self, mode="real", timezone_str="US/Mountain"):
        self.mode = mode
        self.timezone = pytz.timezone(timezone_str)
        self.real_start_time = time.time()
        self.sim_start_time = datetime.datetime.now(self.timezone)
        self.acceleration_factor = 1

    def switch_to_simulate_mode(self, acceleration_factor, start_time_str=None):
        self.mode = "simulate"
        self.acceleration_factor = acceleration_factor
        self.real_start_time = (
            time.time()
        )  # Reset the real time reference when simulation starts

        if start_time_str is None:
            self.sim_start_time = datetime.datetime.now(self.timezone)
        else:
            naive_start_time = datetime.datetime.strptime(
                start_time_str, "%Y-%m-%d %H:%M:%S"
            )
            # Localize the naive datetime to account for DST
            self.sim_start_time = self.timezone.localize(
                naive_start_time, is_dst=None
            )  # None allows pytz to infer DST

    def get_current_time(self):
        if self.mode == "real":
            return datetime.datetime.now(self.timezone)
        else:
            elapsed_real_time = time.time() - self.real_start_time
            elapsed_sim_time = elapsed_real_time * self.acceleration_factor
            current_sim_time = self.sim_start_time + datetime.timedelta(
                seconds=elapsed_sim_time
            )
            # Ensure the simulated time keeps the correct DST settings
            return current_sim_time.astimezone(
                self.timezone
            )  # Normalizes the DST if there has been a change

    def display_time(self):
        try:
            while True:
                sys.stdout.write(f"Simulated time: {self.get_current_time()}\r")
                sys.stdout.flush()
                time.sleep(1)  # Refresh rate of 1 second
        except KeyboardInterrupt:
            print("\nSimulation stopped.")
            sys.stdout.flush()
