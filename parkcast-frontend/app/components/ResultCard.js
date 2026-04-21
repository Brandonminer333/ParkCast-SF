'use client';

import { useState, useEffect } from 'react';

const DEMAND_COLORS = {
  'Low':       { bg: 'bg-green-900/30',  border: 'border-green-700',  text: 'text-green-400',  bar: 'bg-green-500' },
  'Medium':    { bg: 'bg-yellow-900/30', border: 'border-yellow-700', text: 'text-yellow-400', bar: 'bg-yellow-500' },
  'High':      { bg: 'bg-orange-900/30', border: 'border-orange-700', text: 'text-orange-400', bar: 'bg-orange-500' },
  'Very High': { bg: 'bg-red-900/30',    border: 'border-red-700',    text: 'text-red-400',    bar: 'bg-red-500' },
};

function useCountUp(target, duration = 600) {
  const [value, setValue] = useState(0);
  useEffect(() => {
    let start = 0;
    const startTime = performance.now();
    function tick(now) {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setValue(Math.round(eased * target * 100) / 100);
      if (progress < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }, [target, duration]);
  return value;
}

export default function ResultCard({ prediction }) {
  const colors = DEMAND_COLORS[prediction.demand_level] || DEMAND_COLORS['Medium'];
  const occupancy = prediction.predicted_occupancy_pct;
  const animatedOccupancy = useCountUp(occupancy);
  const animatedSpaces = useCountUp(prediction.available_spaces_estimate, 400);

  return (
    <div className="space-y-4">
      {/* Main occupancy display */}
      <div className={`rounded-lg border p-4 ${colors.bg} ${colors.border}`}>
        <div className="flex items-end justify-between mb-3">
          <div>
            <p className="text-xs text-gray-400 mb-1">Predicted occupancy</p>
            <p className={`text-4xl font-black ${colors.text}`}>{animatedOccupancy}%</p>
          </div>
          <div className="text-right">
            <p className="text-xs text-gray-400 mb-1">Demand level</p>
            <span className={`text-sm font-semibold ${colors.text}`}>{prediction.demand_level}</span>
          </div>
        </div>

        {/* Progress bar */}
        <div className="w-full bg-gray-800 rounded-full h-2">
          <div className={`h-2 rounded-full transition-all duration-700 ease-out ${colors.bar}`}
            style={{ width: `${animatedOccupancy}%` }} />
        </div>
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-gray-800 rounded-lg p-3">
          <p className="text-xs text-gray-400 mb-1">Available spaces</p>
          <p className="text-2xl font-bold text-white">{Math.round(animatedSpaces)}</p>
        </div>
        <div className="bg-gray-800 rounded-lg p-3">
          <p className="text-xs text-gray-400 mb-1">Neighborhood</p>
          <p className="text-sm font-semibold text-white capitalize">{prediction.neighborhood}</p>
          <p className="text-xs text-gray-500">{prediction.hour === 0 ? 12 : prediction.hour > 12 ? prediction.hour - 12 : prediction.hour}:00 {prediction.hour < 12 ? 'AM' : 'PM'}</p>
        </div>
      </div>

      {/* Recommendation */}
      <div className="bg-gray-800 rounded-lg p-3">
        <p className="text-xs text-gray-400 mb-1">Recommendation</p>
        <p className="text-sm text-white">{prediction.recommendation}</p>
      </div>
    </div>
  );
}
