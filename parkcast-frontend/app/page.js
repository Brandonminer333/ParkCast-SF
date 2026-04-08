'use client';

import { useState, useEffect } from 'react';
import PredictionForm from './components/PredictionForm';
import ResultCard from './components/ResultCard';
import HistoryTable from './components/HistoryTable';
import WeatherBar from './components/WeatherBar';

export default function Home() {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [apiStatus, setApiStatus] = useState(null);
  const [weather, setWeather] = useState(null);
  const [formWeather, setFormWeather] = useState({ temperature: 62, is_raining: false });

  const API_URL = '';  // empty string — uses Next.js proxy

  // Check API health and fetch weather on load
  useEffect(() => {
    fetch(`/api/health`)
      .then(res => res.json())
      .then(data => setApiStatus(data))
      .catch(() => setApiStatus({ status: 'unavailable' }));

    fetch('https://api.open-meteo.com/v1/forecast?latitude=37.7749&longitude=-122.4194&current=temperature_2m,rain,weather_code&temperature_unit=fahrenheit')
      .then(res => res.json())
      .then(data => {
        const w = data.current;
        setWeather(w);
        const code = w.weather_code;
        setFormWeather({
          temperature: Math.round(w.temperature_2m),
          is_raining: w.rain > 0 || (code >= 51 && code <= 82),
        });
      })
      .catch(() => setWeather(null));
  }, []);

  const handlePredict = async (formData) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });

      if (!response.ok) throw new Error('Prediction failed');

      const result = await response.json();
      setPrediction(result);

      // Add to history
      const historyEntry = {
        id: Date.now(),
        timestamp: new Date().toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true }),
        neighborhood: formData.neighborhood,
        hour: formData.hour,
        occupancy: result.predicted_occupancy_pct,
        demand: result.demand_level,
        available: result.available_spaces_estimate,
      };
      setHistory(prev => [historyEntry, ...prev].slice(0, 10));

    } catch (err) {
      setError('Failed to get prediction. Make sure the API is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gray-950 text-white">
      {/* Header */}
      <header className="border-b border-teal-900 bg-gray-900">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-teal-500 rounded flex items-center justify-center font-black text-gray-900 text-lg">
              P
            </div>
            <div>
              <h1 className="text-xl font-bold text-white tracking-tight">ParkCast SF</h1>
              <p className="text-xs text-teal-400">Parking Prediction Engine</p>
            </div>
          </div>

          {/* API Status */}
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${apiStatus?.status === 'healthy' ? 'bg-teal-400' : 'bg-red-400'}`} />
            <span className="text-xs text-gray-400">
              {apiStatus?.status === 'healthy' ? 'API Online' : 'API Offline'}
            </span>
          </div>
        </div>
      </header>

      <div className="max-w-6xl mx-auto px-6 py-8">
        {/* Hero */}
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-white mb-2">
            Find parking <span className="text-teal-400">before you leave</span>
          </h2>
          <p className="text-gray-400">
            Predict parking occupancy 30–60 minutes ahead using ML trained on SF sensor data.
          </p>
        </div>

        {/* Live Weather */}
        {weather && <WeatherBar weather={weather} formWeather={formWeather} />}

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Form */}
          <div className="bg-gray-900 rounded-xl border border-gray-800 p-6">
            <h3 className="text-lg font-semibold mb-4 text-white">Predict Parking</h3>
            <PredictionForm key={weather ? 'loaded' : 'init'} onSubmit={handlePredict} loading={loading} weather={weather} onWeatherChange={setFormWeather} />
          </div>

          {/* Result */}
          <div className="bg-gray-900 rounded-xl border border-gray-800 p-6">
            <h3 className="text-lg font-semibold mb-4 text-white">Prediction Result</h3>
            {error && (
              <div className="bg-red-900/30 border border-red-700 rounded-lg p-4 text-red-300 text-sm">
                {error}
              </div>
            )}
            {loading && (
              <div className="flex items-center justify-center h-40">
                <div className="w-8 h-8 border-2 border-teal-400 border-t-transparent rounded-full animate-spin" />
              </div>
            )}
            {!loading && !error && prediction && <ResultCard prediction={prediction} />}
            {!loading && !error && !prediction && (
              <div className="flex flex-col items-center justify-center h-52 text-gray-600">
                <svg className="w-16 h-16 mb-4 text-gray-700" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="3" y="6" width="18" height="13" rx="2" />
                  <path d="M3 10h18" />
                  <circle cx="7.5" cy="15" r="1.5" />
                  <circle cx="16.5" cy="15" r="1.5" />
                  <path d="M5 6l1.5-3h11L19 6" />
                </svg>
                <p className="text-sm font-medium text-gray-500">No prediction yet</p>
                <p className="text-xs text-gray-600 mt-1">Choose a neighborhood and time, then hit Predict</p>
              </div>
            )}
          </div>
        </div>

        {/* History */}
        {history.length > 0 && (
          <div className="bg-gray-900 rounded-xl border border-gray-800 p-6">
            <h3 className="text-lg font-semibold mb-4 text-white">Prediction History</h3>
            <HistoryTable history={history} />
          </div>
        )}
      </div>
    </main>
  );
}
