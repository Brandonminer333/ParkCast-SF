'use client';

const DEMAND_COLORS = {
  'Low':       { bg: 'bg-green-900/30',  border: 'border-green-700',  text: 'text-green-400',  bar: 'bg-green-500' },
  'Medium':    { bg: 'bg-yellow-900/30', border: 'border-yellow-700', text: 'text-yellow-400', bar: 'bg-yellow-500' },
  'High':      { bg: 'bg-orange-900/30', border: 'border-orange-700', text: 'text-orange-400', bar: 'bg-orange-500' },
  'Very High': { bg: 'bg-red-900/30',    border: 'border-red-700',    text: 'text-red-400',    bar: 'bg-red-500' },
};

export default function ResultCard({ prediction }) {
  const colors = DEMAND_COLORS[prediction.demand_level] || DEMAND_COLORS['Medium'];
  const occupancy = prediction.predicted_occupancy_pct;

  return (
    <div className="space-y-4">
      {/* Main occupancy display */}
      <div className={`rounded-lg border p-4 ${colors.bg} ${colors.border}`}>
        <div className="flex items-end justify-between mb-3">
          <div>
            <p className="text-xs text-gray-400 mb-1">Predicted occupancy</p>
            <p className={`text-4xl font-black ${colors.text}`}>{occupancy}%</p>
          </div>
          <div className="text-right">
            <p className="text-xs text-gray-400 mb-1">Demand level</p>
            <span className={`text-sm font-semibold ${colors.text}`}>{prediction.demand_level}</span>
          </div>
        </div>

        {/* Progress bar */}
        <div className="w-full bg-gray-800 rounded-full h-2">
          <div className={`h-2 rounded-full transition-all ${colors.bar}`}
            style={{ width: `${occupancy}%` }} />
        </div>
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-gray-800 rounded-lg p-3">
          <p className="text-xs text-gray-400 mb-1">Available spaces</p>
          <p className="text-2xl font-bold text-white">{prediction.available_spaces_estimate}</p>
        </div>
        <div className="bg-gray-800 rounded-lg p-3">
          <p className="text-xs text-gray-400 mb-1">Neighborhood</p>
          <p className="text-sm font-semibold text-white capitalize">{prediction.neighborhood}</p>
          <p className="text-xs text-gray-500">Hour {prediction.hour}:00</p>
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
