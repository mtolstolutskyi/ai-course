import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT = (3, 7)


class WeatherError(Exception):
    status = 502

    def __init__(self, message):
        super().__init__(message)
        self.message = message


class CityNotFound(WeatherError):
    status = 404


class BadRequest(WeatherError):
    status = 400


class ServiceUnavailable(WeatherError):
    status = 503


class BadResponse(WeatherError):
    status = 502


def _get_json(url, params):
    try:
        response = requests.get(url, params=params, timeout=TIMEOUT)
    except requests.Timeout:
        raise ServiceUnavailable("Сервіс погоди не відповів вчасно, спробуйте пізніше")
    except requests.ConnectionError:
        raise ServiceUnavailable("Немає зв'язку з сервісом погоди")
    except requests.RequestException:
        raise ServiceUnavailable("Не вдалося звернутися до сервісу погоди")

    if 400 <= response.status_code < 500:
        raise BadResponse(f"Сервіс відхилив запит (код {response.status_code})")
    if response.status_code >= 500:
        raise ServiceUnavailable(f"Сервіс погоди тимчасово недоступний (код {response.status_code})")

    try:
        data = response.json()
    except ValueError:
        raise BadResponse("Сервіс повернув відповідь не у форматі JSON")
    if not isinstance(data, dict):
        raise BadResponse("Неочікуваний формат відповіді сервісу")
    return data


def find_city(name):
    name = (name or "").strip()
    if not name:
        raise BadRequest("Введіть назву міста")
    if len(name) > 100:
        raise BadRequest("Занадто довга назва міста")

    data = _get_json(GEOCODING_URL, {"name": name, "count": 1, "language": "uk", "format": "json"})
    results = data.get("results")
    if not results:
        raise CityNotFound(f"Місто «{name}» не знайдено")

    place = results[0]
    try:
        return {
            "name": place["name"],
            "country": place.get("country", ""),
            "latitude": float(place["latitude"]),
            "longitude": float(place["longitude"]),
        }
    except (KeyError, TypeError, ValueError):
        raise BadResponse("У відповіді геокодування немає координат міста")


def get_current_weather(city):
    place = find_city(city)
    data = _get_json(FORECAST_URL, {
        "latitude": place["latitude"],
        "longitude": place["longitude"],
        "current": "temperature_2m,wind_speed_10m",
        "wind_speed_unit": "ms",
        "timezone": "auto",
    })

    current = data.get("current")
    if not isinstance(current, dict):
        raise BadResponse("У відповіді немає поточної погоди")
    try:
        temperature = float(current["temperature_2m"])
        wind = float(current["wind_speed_10m"])
    except (KeyError, TypeError, ValueError):
        raise BadResponse("У відповіді немає температури або швидкості вітру")

    return {
        "city": place["name"],
        "country": place["country"],
        "latitude": place["latitude"],
        "longitude": place["longitude"],
        "temperature": temperature,
        "wind_speed": wind,
        "time": current.get("time"),
    }
