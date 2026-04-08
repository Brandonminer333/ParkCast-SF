'use client';

const WEATHER_DESCRIPTIONS = {
  0: 'Clear sky',
  1: 'Mainly clear', 2: 'Partly cloudy', 3: 'Overcast',
  45: 'Foggy', 48: 'Icing fog',
  51: 'Light drizzle', 53: 'Drizzle', 55: 'Heavy drizzle',
  61: 'Light rain', 63: 'Rain', 65: 'Heavy rain',
  71: 'Light snow', 73: 'Snow', 75: 'Heavy snow',
  80: 'Light showers', 81: 'Showers', 82: 'Heavy showers',
  95: 'Thunderstorm',
};

const WEATHER_ICONS = {
  0: '☀️', 1: '🌤️', 2: '⛅', 3: '☁️',
  45: '🌫️', 48: '🌫️',
  51: '🌦️', 53: '🌧️', 55: '🌧️',
  61: '🌦️', 63: '🌧️', 65: '🌧️',
  71: '🌨️', 73: '🌨️', 75: '🌨️',
  80: '🌦️', 81: '🌧️', 82: '🌧️',
  95: '⛈️',
};

export default function WeatherBar({ weather, formWeather }) {
  const code = weather.weather_code;
  const temp = formWeather.temperature;
  const isRaining = formWeather.is_raining;
  const desc = WEATHER_DESCRIPTIONS[code] || 'Unknown';
  const icon = isRaining ? '🌧️' : (WEATHER_ICONS[code] || '🌡️');

  const liveTemp = Math.round(weather.temperature_2m);
  const isOverridden = temp !== liveTemp || isRaining !== (weather.rain > 0 || (code >= 51 && code <= 82));

  return (
    <div className="flex items-center gap-4 bg-gray-900 rounded-xl border border-gray-800 px-5 py-3 mb-6">
      <span className="text-2xl">{icon}</span>
      <div className="flex items-center gap-4 text-sm">
        <span className="text-white font-semibold">{temp}°F</span>
        <span className="text-gray-400">{desc}</span>
        {isRaining && (
          <span className="text-blue-400 font-medium">Raining</span>
        )}
      </div>
      <span className="text-xs text-gray-600 ml-auto">
        {isOverridden ? 'Modified from live' : 'Live from SF'}
      </span>
    </div>
  );
}
