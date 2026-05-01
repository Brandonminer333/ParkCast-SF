/**
 * TimePicker — "Leave now" / "Leave later" toggle with minute slider.
 */
'use client';

export default function TimePicker({ show, onToggle, minutesAway, onMinutesChange, hour }) {
  return (
    <div style={{ padding: '12px 16px', borderBottom: '1px solid #1e3a52' }}>
      <button onClick={onToggle}
        style={{
          width: '100%', padding: '8px 14px', borderRadius: 7,
          border: '1px solid #1e3a52', background: show ? '#112031' : 'transparent',
          color: show ? '#14b8a6' : '#64748b', fontSize: 12, cursor: 'pointer',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        }}>
        <span>{show ? `Leaving in ${minutesAway} min (arriving ~${(hour + Math.floor(minutesAway/60)) % 24}:${String(minutesAway % 60).padStart(2,'0')})` : 'Leaving now'}</span>
        <span style={{ fontSize: 10 }}>{show ? '▲' : '▼ Plan ahead'}</span>
      </button>

      {show && (
        <div style={{ marginTop: 10, padding: '12px', background: '#112031', borderRadius: 8 }}>
          <div style={{ fontSize: 11, color: '#64748b', marginBottom: 6 }}>Leaving in: <b style={{ color: '#14b8a6' }}>{minutesAway} minutes</b></div>
          <input type="range" min={0} max={120} step={5} value={minutesAway}
            onChange={e => onMinutesChange(+e.target.value)}
            style={{ width: '100%', accentColor: '#0d9488' }} />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#475569', marginTop: 4 }}>
            <span>Now</span><span>30 min</span><span>1 hour</span><span>2 hours</span>
          </div>
        </div>
      )}
    </div>
  );
}
