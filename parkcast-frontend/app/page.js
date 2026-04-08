'use client';

import { useState, useEffect } from 'react';
import PredictionForm from './components/PredictionForm';
import ResultCard from './components/ResultCard';
import HistoryTable from './components/HistoryTable';

export default function Home() {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [apiStatus, setApiStatus] = useState(null);

  const API_URL = '';  // empty string — uses Next.js proxy

  // Check API health on load
  useEffect(() => {
    fetch(`/api/health`)
      .then(res => res.json())
      .then(data => setApiStatus(data))
      .catch(() => setApiStatus({ status: 'unavailable' }));
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

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Form */}
          <div className="bg-gray-900 rounded-xl border border-gray-800 p-6">
            <h3 className="text-lg font-semibold mb-4 text-white">Predict Parking</h3>
            <PredictionForm onSubmit={handlePredict} loading={loading} />
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
              <div className="flex items-center justify-center h-40 text-gray-600 text-sm">
                Fill in the form to get a prediction
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
