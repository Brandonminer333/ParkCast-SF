/**
 * SearchBar — destination search with offline autocomplete.
 */
'use client';

export default function SearchBar({ query, onQueryChange, suggestions, showSuggestions, onShowSuggestions, onSelect, isRaining }) {
  return (
    <div style={{ padding: '14px 16px', borderBottom: '1px solid #1e3a52' }}>
      <div style={{ fontSize: 11, color: '#64748b', marginBottom: 8, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Where are you going?</div>
      <div style={{ position: 'relative' }}>
        <input
          value={query}
          onChange={e => onQueryChange(e.target.value)}
          onFocus={() => suggestions.length > 0 && onShowSuggestions(true)}
          placeholder="Search address or place..."
          style={{
            width: '100%', background: '#112031', border: '1px solid #1e3a52',
            borderRadius: 8, padding: '10px 14px', color: 'white', fontSize: 13,
            outline: 'none', boxSizing: 'border-box',
          }}
        />
        {showSuggestions && suggestions.length > 0 && (
          <div style={{
            position: 'absolute', top: '100%', left: 0, right: 0, zIndex: 9999,
            background: '#112031', border: '1px solid #1e3a52', borderRadius: 8,
            marginTop: 4, overflow: 'hidden', boxShadow: '0 8px 24px rgba(0,0,0,0.4)',
          }}>
            {suggestions.map((s, i) => (
              <div key={i} onClick={() => onSelect(s)}
                style={{
                  padding: '10px 14px', cursor: 'pointer', fontSize: 12,
                  borderBottom: i < suggestions.length - 1 ? '1px solid #1e3a52' : 'none',
                  color: '#cbd5e1',
                }}
                onMouseEnter={e => e.currentTarget.style.background = '#1e3a52'}
                onMouseLeave={e => e.currentTarget.style.background = 'transparent'}>
                <div style={{ fontWeight: 600, color: 'white', marginBottom: 2 }}>
                  {s._name || s.display_name.split(',')[0]}
                </div>
                <div style={{ color: '#64748b', fontSize: 11 }}>
                  {s.display_name.split(',').slice(1, 3).join(',')}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {isRaining && (
        <div style={{ marginTop: 8, display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          <span style={{ fontSize: 11, padding: '3px 8px', background: '#1e3a52', borderRadius: 4, color: '#94a3b8' }}>
            🌧 Raining detected
          </span>
        </div>
      )}
    </div>
  );
}
