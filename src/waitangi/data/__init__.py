"""Data ingestion and management."""

from waitangi.data.gauge import GaugeData, GaugeReading, fetch_nrc_gauge_data
from waitangi.data.rainfall import RainfallData, RainfallReading, fetch_rainfall_data
from waitangi.data.tide import TidePrediction, fetch_tide_predictions
from waitangi.data.weather import WeatherData, WeatherForecast, fetch_weather_forecast

__all__ = [
    "GaugeData",
    "GaugeReading",
    "RainfallData",
    "RainfallReading",
    "TidePrediction",
    "WeatherData",
    "WeatherForecast",
    "fetch_nrc_gauge_data",
    "fetch_rainfall_data",
    "fetch_tide_predictions",
    "fetch_weather_forecast",
]
