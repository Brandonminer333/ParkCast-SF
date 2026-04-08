'use client';

import { useState } from 'react';

const NEIGHBORHOODS = [
  { value: 'mission', label: 'Mission' },
  { value: 'soma', label: 'SoMa' },
  { value: 'castro', label: 'Castro' },
  { value: 'marina', label: 'Marina' },
  { value: 'tenderloin', label: 'Tenderloin' },
  { value: 'haight', label: 'Haight' },
  { value: 'noe valley', label: 'Noe Valley' },
  { value: 'sunset', label: 'Sunset' },
  { value: 'richmond', label: 'Richmond' },
];

const MONTHS = [
  { value: 1, label: 'January' },
  { value: 2, label: 'February' },
  { value: 3, label: 'March' },
  { value: 4, label: 'April' },
  { value: 5, label: 'May' },
  { value: 6, label: 'June' },
  { value: 7, label: 'July' },
  { value: 8, label: 'August' },
  { value: 9, label: 'September' },
  { value: 10, label: 'October' },
  { value: 11, label: 'November' },
  { value: 12, label: 'December' },
];

const DAYS = [
  { value: 0, label: 'Monday' },
  { value: 1, label: 'Tuesday' },
  { value: 2, label: 'Wednesday' },
  { value: 3, label: 'Thursday' },
  { value: 4, label: 'Friday' },
  { value: 5, label: 'Saturday' },
  { value: 6, label: 'Sunday' },
];

export default function PredictionForm({ onSubmit, loading }) {
  const now = new Date();
  const [form, setForm] = useState({
    neighborhood: 'mission',
    hour: now.getHours(),
    day_of_week: now.getDay() === 0 ? 6 : now.getDay() - 1,
    month: now.getMonth() + 1,
    total_spaces: 40,
    is_raining: 0,
    has_nearby_event: 0,
    is_holiday: 0,
    is_school_day: 1,
    temperature: 62,
  });

  const handleChange = (e) => {
    const { name, value, type } = e.target;
    setForm(prev => ({
      ...prev,
      [name]: type === 'number' || type === 'range' ? Number(value) : value,
    }));
  };

  const handleToggle = (name) => {
    setForm(prev => ({ ...prev, [name]: prev[name] === 1 ? 0 : 1 }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(form);
  };

  const inputClass = "w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-teal-500 transition-colors";
  const labelClass = "block text-xs text-gray-400 mb-1";

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* Neighborhood */}
      <div>
        <label className={labelClass}>Neighborhood</label>
        <select name="neighborhood" value={form.neighborhood} onChange={handleChange} className={inputClass}>
          {NEIGHBORHOODS.map(n => (
            <option key={n.value} value={n.value}>{n.label}</option>
          ))}
        </select>
      </div>

      {/* Hour + Day */}
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className={labelClass}>Hour: {form.hour === 0 ? 12 : form.hour > 12 ? form.hour - 12 : form.hour}:00 {form.hour < 12 ? 'AM' : 'PM'}</label>
          <input type="range" name="hour" min="0" max="23" value={form.hour}
            onChange={handleChange} className="w-full accent-teal-500" />
        </div>
        <div>
          <label className={labelClass}>Day of week</label>
          <select name="day_of_week" value={form.day_of_week} onChange={handleChange} className={inputClass}>
            {DAYS.map(d => (
              <option key={d.value} value={d.value}>{d.label}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Month + Total Spaces */}
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className={labelClass}>Month</label>
          <select name="month" value={form.month} onChange={handleChange} className={inputClass}>
            {MONTHS.map(m => (
              <option key={m.value} value={m.value}>{m.label}</option>
            ))}
          </select>
        </div>
        <div>
          <label className={labelClass}>Total spaces</label>
          <input type="number" name="total_spaces" min="1" max="200" value={form.total_spaces}
            onChange={handleChange} className={inputClass} />
        </div>
      </div>

      {/* Temperature */}
      <div>
        <label className={labelClass}>Temperature: {form.temperature}°F</label>
        <input type="range" name="temperature" min="40" max="90" value={form.temperature}
          onChange={handleChange} className="w-full accent-teal-500" />
      </div>

      {/* Toggles */}
      <div className="grid grid-cols-2 gap-2">
        {[
          { key: 'is_raining', label: 'Raining' },
          { key: 'has_nearby_event', label: 'Nearby Event' },
          { key: 'is_holiday', label: 'Holiday' },
          { key: 'is_school_day', label: 'School Day' },
        ].map(({ key, label }) => (
          <button key={key} type="button" onClick={() => handleToggle(key)}
            className={`flex items-center justify-between px-3 py-2 rounded-lg border text-sm transition-colors ${
              form[key] === 1
                ? 'bg-teal-900/40 border-teal-600 text-teal-300'
                : 'bg-gray-800 border-gray-700 text-gray-400'
            }`}>
            <span>{label}</span>
            <div className={`w-8 h-4 rounded-full transition-colors relative ${form[key] === 1 ? 'bg-teal-500' : 'bg-gray-600'}`}>
              <div className={`absolute top-0.5 w-3 h-3 rounded-full bg-white transition-transform ${form[key] === 1 ? 'translate-x-4' : 'translate-x-0.5'}`} />
            </div>
          </button>
        ))}
      </div>

      {/* Submit */}
      <button type="submit" disabled={loading}
        className="w-full bg-teal-500 hover:bg-teal-400 disabled:bg-teal-900 disabled:text-teal-700 text-gray-900 font-semibold py-2.5 rounded-lg transition-colors text-sm">
        {loading ? 'Predicting...' : 'Predict Parking'}
      </button>
    </form>
  );
}
