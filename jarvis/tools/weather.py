"""Weather via Open-Meteo (free, no API key)."""

import httpx

from jarvis.tools.registry import Tool, params

_GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST = "https://api.open-meteo.com/v1/forecast"

# WMO weather interpretation codes → words
_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light showers", 81: "showers", 82: "violent showers",
    85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}


def describe(code: int | None) -> str:
    return _CODES.get(code, "unknown conditions") if code is not None else "unknown conditions"


def geocode(place: str, client: httpx.Client) -> dict | None:
    name = place.split(",")[0].strip()
    r = client.get(_GEOCODE, params={"name": name, "count": 5, "language": "en", "format": "json"})
    r.raise_for_status()
    results = r.json().get("results") or []
    if not results:
        return None
    # If the user gave a country/region after a comma, prefer a matching result.
    hint = place.split(",", 1)[1].strip().lower() if "," in place else ""
    for res in results:
        if hint and hint in f"{res.get('country', '')} {res.get('admin1', '')}".lower():
            return res
    return results[0]


def format_report(place: dict, data: dict) -> str:
    cur, daily = data["current"], data["daily"]
    where = ", ".join(p for p in (place.get("name"), place.get("admin1"), place.get("country")) if p)
    lines = [
        f"{where} now: {describe(cur.get('weather_code'))}, {cur['temperature_2m']:.0f}°C "
        f"(feels like {cur['apparent_temperature']:.0f}°C), humidity {cur['relative_humidity_2m']}%, "
        f"wind {cur['wind_speed_10m']:.0f} km/h."
    ]
    for i, day in enumerate(daily["time"][:3]):
        name = ["Today", "Tomorrow", day][i]
        lines.append(
            f"{name}: {describe(daily['weather_code'][i])}, {daily['temperature_2m_min'][i]:.0f} to "
            f"{daily['temperature_2m_max'][i]:.0f}°C, {daily['precipitation_probability_max'][i]}% chance of rain."
        )
    return " ".join(lines)


def make_weather_tool(default_location: str) -> Tool:
    def get_weather(location: str = "") -> str:
        place_name = location.strip() or default_location
        if not place_name:
            return "Error: no location given and no home location configured. Ask the user which city."
        with httpx.Client(timeout=10) as client:
            place = geocode(place_name, client)
            if place is None:
                return f"Error: couldn't find a place called '{place_name}'."
            r = client.get(
                _FORECAST,
                params={
                    "latitude": place["latitude"],
                    "longitude": place["longitude"],
                    "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
                    "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                    "timezone": "auto",
                    "forecast_days": 3,
                },
            )
            r.raise_for_status()
            return format_report(place, r.json())

    home = f" Defaults to the user's home, {default_location}." if default_location else ""
    return Tool(
        "get_weather",
        f"Current weather and 3-day forecast for a place.{home}",
        params(location={"type": "string", "description": "City, optionally ', Country'"}),
        get_weather,
    )
