from typing import Tuple
# This file uses GitHub Copilot for code assistance.


import torch


class WeatherForecast:
    def __init__(self, data_raw: list[list[float]]):
        """
        You are given a list of 10 weather measurements per day.
        Save the data as a PyTorch (num_days, 10) tensor,
        where the first dimension represents the day,
        and the second dimension represents the measurements.
        """
        self.data = torch.as_tensor(data_raw).view(-1, 10)

    def find_min_and_max_per_day(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find the max and min temperatures per day

        Returns:
            min_per_day: tensor of size (num_days,)
            max_per_day: tensor of size (num_days,)
        """
        min_per_day = torch.min(self.data, dim=1).values
        max_per_day = torch.max(self.data, dim=1).values
        return min_per_day, max_per_day

    def find_the_largest_drop(self) -> torch.Tensor:
        """
        Find the largest change in day over day average temperature.
        This should be a negative number.

        Returns:
            tensor of a single value, the difference in temperature
        """
        avg = self.data.mean(dim=1)
        diff = avg[1:] - avg[:-1]
        return torch.min(diff)

    def find_the_most_extreme_day(self) -> torch.Tensor:
        """
        For each day, find the measurement that differs the most from the day's average temperature

        Returns:
            tensor with size (num_days,)
        """
        avg = self.data.mean(dim=1, keepdim=True)
        diff = (self.data - avg).abs()
        idx = torch.argmax(diff, dim=1)
        return self.data[torch.arange(self.data.size(0)), idx]

    def max_last_k_days(self, k: int) -> torch.Tensor:
        """
        Find the maximum temperature over the last k days

        Returns:
            tensor of size (k,)
        """
        return torch.max(self.data[-k:], dim=1).values

    def predict_temperature(self, k: int) -> torch.Tensor:
        """
        From the dataset, predict the temperature of the next day.
        The prediction will be the average of the temperatures over the past k days.

        Args:
            k: int, number of days to consider

        Returns:
            tensor of a single value, the predicted temperature
        """
        return self.data[-k:].mean()

    def what_day_is_this_from(self, t: torch.FloatTensor) -> torch.LongTensor:
        """
        Given a list of 10 temperature measurements, find the day in a week
        that the temperature is most likely measured on.

        Args:
            t: tensor of size (10,), temperature measurements

        Returns:
            tensor of a single value, the index of the closest data element
        """
        diffs = (self.data - t).abs().sum(dim=1)
        return torch.argmin(diffs)
