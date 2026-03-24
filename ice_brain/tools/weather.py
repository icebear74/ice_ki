"""
Wetter-Tool – Echtzeit-Wetterdaten via Open-Meteo API (kein API-Key nötig).

Basiert auf der WeatherModule-Implementierung aus icebear74/Panelclock (C++).

API-Endpunkte:
    Vorhersage:  https://api.open-meteo.com/v1/forecast
    Historisch:  https://archive-api.open-meteo.com/v1/archive

Koordinaten kommen aus:
    1. get_active_location(user_id) aus tools/location.py
    2. Explizit genannter Ortsname → Geocoding via Nominatim
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from . import register_tool

logger = logging.getLogger(__name__)

_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# WMO Weather Interpretation Codes → Deutsche Beschreibungen
_WMO_CODES: dict[int, str] = {
    0: "Klar",
    1: "Überwiegend klar",
    2: "Teilweise bewölkt",
    3: "Bedeckt",
    45: "Nebel",
    48: "Reifnebel",
    51: "Leichter Nieselregen",
    53: "Mäßiger Nieselregen",
    55: "Starker Nieselregen",
    56: "Leichter gefrierender Nieselregen",
    57: "Starker gefrierender Nieselregen",
    61: "Leichter Regen",
    63: "Mäßiger Regen",
    65: "Starker Regen",
    66: "Leichter gefrierender Regen",
    67: "Starker gefrierender Regen",
    71: "Leichter Schneefall",
    73: "Mäßiger Schneefall",
    75: "Starker Schneefall",
    77: "Schneegriesel",
    80: "Leichte Regenschauer",
    81: "Mäßige Regenschauer",
    82: "Starke Regenschauer",
    85: "Leichte Schneeschauer",
    86: "Starke Schneeschauer",
    95: "Gewitter",
    96: "Gewitter mit leichtem Hagel",
    99: "Gewitter mit starkem Hagel",
}


def _wmo_description(code: int | None) -> str:
    """WMO-Wettercodes in deutschen Klartext umwandeln."""
    if code is None:
        return "Unbekannt"
    return _WMO_CODES.get(int(code), f"Code {code}")


def get_current_weather(lat: float, lon: float) -> dict[str, Any]:
    """Aktuelles Wetter für die gegebenen Koordinaten abrufen.

    Gibt ein dict mit den aktuellen Wetterdaten zurück.
    """
    try:
        import httpx  # noqa: PLC0415
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": ",".join([
                "temperature_2m",
                "relative_humidity_2m",
                "apparent_temperature",
                "is_day",
                "precipitation",
                "rain",
                "showers",
                "snowfall",
                "weather_code",
                "cloud_cover",
                "wind_speed_10m",
                "wind_gusts_10m",
                "uv_index",
            ]),
            "timezone": "UTC",
        }
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(_FORECAST_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

        current = data.get("current", {})
        code = current.get("weather_code")
        return {
            "temperature_c": current.get("temperature_2m"),
            "feels_like_c": current.get("apparent_temperature"),
            "humidity_pct": current.get("relative_humidity_2m"),
            "condition": _wmo_description(code),
            "weather_code": code,
            "is_day": bool(current.get("is_day", 1)),
            "precipitation_mm": current.get("precipitation"),
            "rain_mm": current.get("rain"),
            "showers_mm": current.get("showers"),
            "snowfall_cm": current.get("snowfall"),
            "cloud_cover_pct": current.get("cloud_cover"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
            "wind_gusts_kmh": current.get("wind_gusts_10m"),
            "uv_index": current.get("uv_index"),
            "time": current.get("time"),
        }

    except Exception as exc:  # noqa: BLE001
        logger.warning("get_current_weather fehlgeschlagen (lat=%.4f, lon=%.4f): %s", lat, lon, exc)
        return {}


def get_hourly_forecast(lat: float, lon: float, hours: int = 48) -> list[dict[str, Any]]:
    """Stündliche Wettervorhersage für die nächsten *hours* Stunden.

    Gibt eine Liste von Dicts zurück (ein Eintrag pro Stunde).
    """
    try:
        import httpx  # noqa: PLC0415
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join([
                "temperature_2m",
                "apparent_temperature",
                "precipitation_probability",
                "precipitation",
                "rain",
                "snowfall",
                "weather_code",
            ]),
            "forecast_hours": min(hours, 168),  # max 7 Tage = 168h
            "timezone": "UTC",
        }
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(_FORECAST_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        result = []
        for i, t in enumerate(times[:hours]):
            result.append({
                "time": t,
                "temperature_c": _get_idx(hourly, "temperature_2m", i),
                "feels_like_c": _get_idx(hourly, "apparent_temperature", i),
                "precipitation_probability_pct": _get_idx(hourly, "precipitation_probability", i),
                "precipitation_mm": _get_idx(hourly, "precipitation", i),
                "rain_mm": _get_idx(hourly, "rain", i),
                "snowfall_cm": _get_idx(hourly, "snowfall", i),
                "condition": _wmo_description(_get_idx(hourly, "weather_code", i)),
                "weather_code": _get_idx(hourly, "weather_code", i),
            })
        return result

    except Exception as exc:  # noqa: BLE001
        logger.warning("get_hourly_forecast fehlgeschlagen (lat=%.4f, lon=%.4f): %s", lat, lon, exc)
        return []


def get_daily_forecast(lat: float, lon: float, days: int = 7) -> list[dict[str, Any]]:
    """Tägliche Wettervorhersage für die nächsten *days* Tage (max. 14).

    Gibt eine Liste von Dicts zurück (ein Eintrag pro Tag).
    """
    try:
        import httpx  # noqa: PLC0415
        forecast_days = min(max(days, 1), 14)
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": ",".join([
                "weather_code",
                "temperature_2m_max",
                "temperature_2m_min",
                "temperature_2m_mean",
                "sunrise",
                "sunset",
                "precipitation_sum",
                "rain_sum",
                "snowfall_sum",
                "precipitation_probability_max",
                "uv_index_max",
                "cloud_cover_mean",
                "wind_speed_10m_max",
                "sunshine_duration",
            ]),
            "forecast_days": forecast_days,
            "timezone": "UTC",
        }
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(_FORECAST_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

        daily = data.get("daily", {})
        dates = daily.get("time", [])
        result = []
        for i, d in enumerate(dates):
            result.append({
                "date": d,
                "condition": _wmo_description(_get_idx(daily, "weather_code", i)),
                "weather_code": _get_idx(daily, "weather_code", i),
                "temp_max_c": _get_idx(daily, "temperature_2m_max", i),
                "temp_min_c": _get_idx(daily, "temperature_2m_min", i),
                "temp_mean_c": _get_idx(daily, "temperature_2m_mean", i),
                "sunrise": _get_idx(daily, "sunrise", i),
                "sunset": _get_idx(daily, "sunset", i),
                "precipitation_sum_mm": _get_idx(daily, "precipitation_sum", i),
                "rain_sum_mm": _get_idx(daily, "rain_sum", i),
                "snowfall_sum_cm": _get_idx(daily, "snowfall_sum", i),
                "precipitation_probability_max_pct": _get_idx(daily, "precipitation_probability_max", i),
                "uv_index_max": _get_idx(daily, "uv_index_max", i),
                "cloud_cover_mean_pct": _get_idx(daily, "cloud_cover_mean", i),
                "wind_speed_max_kmh": _get_idx(daily, "wind_speed_10m_max", i),
                "sunshine_duration_s": _get_idx(daily, "sunshine_duration", i),
            })
        return result

    except Exception as exc:  # noqa: BLE001
        logger.warning("get_daily_forecast fehlgeschlagen (lat=%.4f, lon=%.4f): %s", lat, lon, exc)
        return []


def get_historical_climate(lat: float, lon: float, years: int = 5) -> dict[str, Any]:
    """Historische Klimadaten (letzte *years* Jahre, gleicher Datumsbereich) abrufen.

    Gibt ein dict mit dem Durchschnitt der täglichen Mitteltemperatur zurück.
    """
    try:
        import httpx  # noqa: PLC0415
        today = datetime.now(tz=timezone.utc).date()
        # Gleicher Zeitraum (heutiger Tag ± 15 Tage) für die letzten N Jahre
        start_of_range = today - timedelta(days=15)
        end_of_range = today + timedelta(days=15)

        all_temps: list[float] = []
        for year_offset in range(1, years + 1):
            try:
                start = start_of_range.replace(year=start_of_range.year - year_offset)
                end = end_of_range.replace(year=end_of_range.year - year_offset)
                params = {
                    "latitude": lat,
                    "longitude": lon,
                    "daily": "temperature_2m_mean",
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat(),
                    "timezone": "UTC",
                }
                with httpx.Client(timeout=10.0) as client:
                    resp = client.get(_ARCHIVE_URL, params=params)
                    resp.raise_for_status()
                    data = resp.json()
                temps = [t for t in (data.get("daily", {}).get("temperature_2m_mean") or []) if t is not None]
                all_temps.extend(temps)
            except Exception as exc_year:  # noqa: BLE001
                logger.debug("Historische Daten für Jahr -%d fehlgeschlagen: %s", year_offset, exc_year)

        if not all_temps:
            return {}

        avg = sum(all_temps) / len(all_temps)
        return {
            "avg_temp_c": round(avg, 1),
            "years": years,
            "samples": len(all_temps),
        }

    except Exception as exc:  # noqa: BLE001
        logger.warning("get_historical_climate fehlgeschlagen (lat=%.4f, lon=%.4f): %s", lat, lon, exc)
        return {}


def _get_idx(data: dict, key: str, idx: int) -> Any:
    """Hilfsfunktion: Holt den Wert an Index *idx* aus data[key], oder None."""
    arr = data.get(key)
    if arr is None or idx >= len(arr):
        return None
    return arr[idx]


def _format_weather_context(
    location_name: str,
    current: dict,
    daily: list[dict],
) -> str:
    """Formatiert die Wetterdaten als lesbaren Text für den System-Prompt."""
    lines: list[str] = [f"🌤️ Wetter für {location_name}:"]

    if current:
        temp = current.get("temperature_c")
        feels = current.get("feels_like_c")
        cond = current.get("condition", "")
        humidity = current.get("humidity_pct")
        wind = current.get("wind_speed_kmh")
        gusts = current.get("wind_gusts_kmh")
        uv = current.get("uv_index")
        precip = current.get("precipitation_mm", 0) or 0

        temp_str = f"{temp:.1f}°C" if temp is not None else "?"
        feels_str = f" (gefühlt {feels:.1f}°C)" if feels is not None else ""
        lines.append(f"  Aktuell: {temp_str}{feels_str}, {cond}")
        if humidity is not None:
            lines.append(f"  Luftfeuchtigkeit: {humidity:.0f}%")
        if wind is not None:
            gusts_str = f", Böen {gusts:.0f} km/h" if gusts else ""
            lines.append(f"  Wind: {wind:.0f} km/h{gusts_str}")
        if precip > 0:
            lines.append(f"  Niederschlag: {precip:.1f} mm")
        if uv is not None:
            lines.append(f"  UV-Index: {uv:.1f}")

    if daily:
        lines.append("  Vorhersage:")
        for day in daily[:5]:
            date = day.get("date", "")
            cond = day.get("condition", "")
            t_max = day.get("temp_max_c")
            t_min = day.get("temp_min_c")
            precip_prob = day.get("precipitation_probability_max_pct")
            t_str = ""
            if t_max is not None and t_min is not None:
                t_str = f" {t_min:.0f}–{t_max:.0f}°C"
            prob_str = f", Regenwahrsch. {precip_prob:.0f}%" if precip_prob else ""
            lines.append(f"    {date}:{t_str}, {cond}{prob_str}")

    return "\n".join(lines)


@register_tool("weather")
def get_weather_for_user(user_id: str = "default", location_name: str | None = None) -> str:
    """Haupteinstiegspunkt: Wetter für den Benutzer oder einen expliziten Ort abrufen.

    Gibt einen formatierten String zurück, der in den System-Prompt oder als
    Tool-Ergebnis eingefügt werden kann.

    Koordinaten kommen aus:
    1. Explizit genannter Ortsname → Geocoding via Nominatim
    2. get_active_location(user_id) aus user_memory
    """
    lat: float | None = None
    lon: float | None = None
    resolved_name = location_name or "Unbekannt"

    # 1. Expliziter Ortsname → Geocoding
    if location_name:
        try:
            from .geocoding import geocode  # noqa: PLC0415
            coords = geocode(location_name)
            if coords:
                lat = coords["lat"]
                lon = coords["lon"]
                resolved_name = coords.get("display_name", location_name).split(",")[0]
            else:
                logger.warning("Geocoding: kein Ergebnis für %r", location_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Geocoding-Fehler für %r: %s", location_name, exc)

    # 2. Fallback: gespeicherter Standort des Benutzers
    if lat is None and user_id != "default":
        try:
            from .location import get_active_location  # noqa: PLC0415
            loc = get_active_location(user_id)
            if loc:
                lat = loc["latitude"]
                lon = loc["longitude"]
                resolved_name = loc.get("content", resolved_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_active_location fehlgeschlagen für user %r: %s", user_id, exc)

    if lat is None or lon is None:
        return ""

    # Wetterdaten abrufen
    current = get_current_weather(lat, lon)
    daily = get_daily_forecast(lat, lon, days=5)

    if not current and not daily:
        return ""

    return _format_weather_context(resolved_name, current, daily)
