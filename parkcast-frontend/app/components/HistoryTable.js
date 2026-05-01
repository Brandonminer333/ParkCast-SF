'use client';

const DEMAND_COLORS = {
  'Low':       'text-green-400',
  'Medium':    'text-yellow-400',
  'High':      'text-orange-400',
  'Very High': 'text-red-400',
};

export default function HistoryTable({ history }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-xs text-gray-500 border-b border-gray-800">
            <th className="text-left pb-2 font-medium">Time</th>
            <th className="text-left pb-2 font-medium">Neighborhood</th>
            <th className="text-left pb-2 font-medium">Hour</th>
            <th className="text-left pb-2 font-medium">Occupancy</th>
            <th className="text-left pb-2 font-medium">Demand</th>
            <th className="text-left pb-2 font-medium">Available</th>
          </tr>
        </thead>
        <tbody>
          {history.map((entry) => (
            <tr key={entry.id} className="border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors">
              <td className="py-2 text-gray-500 text-xs">{entry.timestamp}</td>
              <td className="py-2 text-white capitalize">{entry.neighborhood}</td>
              <td className="py-2 text-gray-300">{entry.hour}:00</td>
              <td className="py-2">
                <div className="flex items-center gap-2">
                  <div className="w-16 bg-gray-800 rounded-full h-1.5">
                    <div className="h-1.5 rounded-full bg-teal-500"
                      style={{ width: `${entry.occupancy}%` }} />
                  </div>
                  <span className="text-white font-medium">{entry.occupancy}%</span>
                </div>
              </td>
              <td className={`py-2 font-medium ${DEMAND_COLORS[entry.demand] || 'text-gray-400'}`}>
                {entry.demand}
              </td>
              <td className="py-2 text-gray-300">{entry.available} spots</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
