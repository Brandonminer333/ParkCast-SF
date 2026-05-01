'use client';

import { useState, useEffect, useRef } from 'react';
import SF_LOCATIONS from './data/sfLocations';
import SearchBar from './components/SearchBar';
import TimePicker from './components/TimePicker';
import BlockList from './components/BlockList';

export default function MapPage() {
  const mapRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const markersRef = useRef([]);
  const destMarkerRef = useRef(null);
  const userMarkerRef = useRef(null);
  const circleRef = useRef(null);
  const driveRouteRef = useRef(null);
  const walkRouteRef = useRef(null);

  const now = new Date();
  const [query, setQuery] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [destination, setDestination] = useState(null);
  const [userLocation, setUserLocation] = useState(null);
  const [leafletLoaded, setLeafletLoaded] = useState(false);
  const [blocks, setBlocks] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedBlock, setSelectedBlock] = useState(null);
  const [routeInfo, setRouteInfo] = useState(null);
  const [driveSteps, setDriveSteps] = useState([]);
  const [bestBlockIndex, setBestBlockIndex] = useState(0);
  const [apiStatus, setApiStatus] = useState(null);
  const [showTimePicker, setShowTimePicker] = useState(false);
  const [hour, setHour] = useState(now.getHours());
  const [minutesAway, setMinutesAway] = useState(0);
  const [conditions, setConditions] = useState({
    is_raining: 0, event_intensity: 0,
    is_holiday: 0, is_school_day: 1,
    temperature: 62, month: now.getMonth() + 1,
    day_of_week: now.getDay() === 0 ? 6 : now.getDay() - 1,
  });

  // ── Auto-fetch weather ────────────────────────────────────────
  useEffect(() => {
    fetch('https://api.open-meteo.com/v1/forecast?latitude=37.7749&longitude=-122.4194&current=temperature_2m,precipitation,weathercode&temperature_unit=fahrenheit&timezone=America%2FLos_Angeles')
      .then(r => r.json())
      .then(data => {
        const current = data.current || {};
        setConditions(prev => ({
          ...prev,
          temperature:  current.temperature_2m || 62,
          is_raining:   (current.precipitation || 0) > 0.1 ? 1 : 0,
        }));
      })
      .catch(() => {});
  }, []);

  // ── User geolocation ──────────────────────────────────────────
  useEffect(() => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (pos) => setUserLocation({ lat: pos.coords.latitude, lon: pos.coords.longitude }),
        () => setUserLocation({ lat: 37.7749, lon: -122.4194 })
      );
    }
  }, []);

  // ── Leaflet loading ───────────────────────────────────────────
  useEffect(() => {
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';
    document.head.appendChild(link);
    const script = document.createElement('script');
    script.src = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';
    script.onload = () => setLeafletLoaded(true);
    document.head.appendChild(script);
  }, []);

  // ── Map init ──────────────────────────────────────────────────
  useEffect(() => {
    if (!leafletLoaded || !mapRef.current || mapInstanceRef.current) return;
    const L = window.L;
    const map = L.map(mapRef.current).setView([37.7749, -122.4194], 14);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap contributors', maxZoom: 19,
    }).addTo(map);
    mapInstanceRef.current = map;
  }, [leafletLoaded]);

  // ── User location marker ──────────────────────────────────────
  useEffect(() => {
    if (!mapInstanceRef.current || !leafletLoaded || !userLocation) return;
    const L = window.L;
    if (userMarkerRef.current) userMarkerRef.current.remove();
    const icon = L.divIcon({
      html: `<div style="width:16px;height:16px;border-radius:50%;background:#3b82f6;border:3px solid white;box-shadow:0 0 0 3px rgba(59,130,246,0.3)"></div>`,
      iconSize: [16, 16], iconAnchor: [8, 8], className: '',
    });
    userMarkerRef.current = L.marker([userLocation.lat, userLocation.lon], { icon })
      .addTo(mapInstanceRef.current)
      .bindPopup('Your location');
  }, [userLocation, leafletLoaded]);

  // ── Destination marker + radius circle ────────────────────────
  useEffect(() => {
    if (!mapInstanceRef.current || !leafletLoaded || !destination) return;
    const L = window.L;
    if (destMarkerRef.current) destMarkerRef.current.remove();
    if (circleRef.current) circleRef.current.remove();
    const icon = L.divIcon({
      html: `<div style="width:20px;height:20px;border-radius:50%;background:#0d9488;border:3px solid white;box-shadow:0 2px 8px rgba(0,0,0,0.4)"></div>`,
      iconSize: [20, 20], iconAnchor: [10, 10], className: '',
    });
    destMarkerRef.current = L.marker([destination.lat, destination.lon], { icon })
      .addTo(mapInstanceRef.current)
      .bindPopup(`<b>${destination.name}</b>`);
    circleRef.current = L.circle([destination.lat, destination.lon], {
      radius: 600, color: '#0d9488', fillColor: '#0d9488',
      fillOpacity: 0.05, weight: 1, dashArray: '4 4',
    }).addTo(mapInstanceRef.current);
    mapInstanceRef.current.setView([destination.lat, destination.lon], 15);
  }, [destination, leafletLoaded]);

  // ── Block markers ─────────────────────────────────────────────
  useEffect(() => {
    if (!mapInstanceRef.current || !leafletLoaded) return;
    const L = window.L;
    markersRef.current.forEach(m => m.remove());
    markersRef.current = [];
    blocks.forEach(block => {
      const icon = L.divIcon({
        html: `<div style="
          background:${block.color};border:2px solid white;
          border-radius:50%;width:40px;height:40px;
          display:flex;align-items:center;justify-content:center;
          font-size:12px;font-weight:700;
          color:white;box-shadow:0 2px 6px rgba(0,0,0,0.35);cursor:pointer;
        ">${Math.round(block.predicted_occupancy_pct)}%</div>`,
        className: '', iconAnchor: [20, 20],
      });
      const marker = L.marker([block.lat, block.lon], { icon })
        .addTo(mapInstanceRef.current)
        .bindPopup(`
          <div style="font-family:sans-serif;min-width:210px;padding:4px">
            <b style="font-size:13px">${block.street}</b><br>
            <span style="color:${block.color};font-size:20px;font-weight:700">${block.predicted_occupancy_pct}%</span>
            <span style="color:#666;font-size:13px"> occupied</span><br>
            <span style="color:#333;font-size:13px">${block.demand_level} demand</span><br>
            <span style="color:#888;font-size:11px">${(block.distance_meters / 1609.344).toFixed(2)} mi from destination</span>
          </div>
        `);
      marker.on('click', () => setSelectedBlock(block));
      markersRef.current.push(marker);
    });
  }, [blocks, leafletLoaded]);

  // ── Clear routes when blocks change ───────────────────────────
  useEffect(() => {
    if (driveRouteRef.current) driveRouteRef.current.remove();
    if (walkRouteRef.current) walkRouteRef.current.remove();
    setRouteInfo(null);
    setDriveSteps([]);
    setSelectedBlock(null);
    setBestBlockIndex(0);
  }, [blocks]);

  // ── API health check ──────────────────────────────────────────
  useEffect(() => {
    fetch('/api/health').then(r => r.json()).then(setApiStatus).catch(() => setApiStatus({ status: 'unavailable' }));
  }, []);

  // ── Route drawing ─────────────────────────────────────────────
  const drawBlockRoutes = async (block) => {
    if (!mapInstanceRef.current || !leafletLoaded) return;
    const L = window.L;
    if (driveRouteRef.current) driveRouteRef.current.remove();
    if (walkRouteRef.current) walkRouteRef.current.remove();
    setRouteInfo(null);
    setDriveSteps([]);

    if (userLocation) {
      try {
        const driveUrl = `/api/route?profile=driving-traffic&coords=${userLocation.lon},${userLocation.lat};${block.lon},${block.lat}`;
        const driveRes = await fetch(driveUrl);
        const driveData = await driveRes.json();
        if (driveData.routes?.[0]) {
          const coords = driveData.routes[0].geometry.coordinates.map(([lon, lat]) => [lat, lon]);
          driveRouteRef.current = L.polyline(coords, { color: '#0d9488', weight: 5, opacity: 0.85 }).addTo(mapInstanceRef.current);
          const driveMin = Math.ceil(driveData.routes[0].duration / 60);
          setDriveSteps(driveData.routes[0].legs?.[0]?.steps || []);
          const driveMi = driveData.routes[0].distance / 1609.344;

          let walkMin = null;
          if (destination) {
            try {
              const walkUrl = `/api/route?profile=walking&coords=${block.lon},${block.lat};${destination.lon},${destination.lat}`;
              const walkRes = await fetch(walkUrl);
              const walkData = await walkRes.json();
              if (walkData.routes?.[0]) {
                const walkCoords = walkData.routes[0].geometry.coordinates.map(([lon, lat]) => [lat, lon]);
                walkRouteRef.current = L.polyline(walkCoords, { color: '#ffffff', weight: 3, opacity: 0.8, dashArray: '6 5' }).addTo(mapInstanceRef.current);
                walkMin = Math.ceil(walkData.routes[0].duration / 60);
              }
            } catch { /* walk route is best-effort */ }
          }
          setRouteInfo({ driveMin, walkMin, driveMi });
        }
      } catch { /* drive route is best-effort */ }
    }

    const map = mapInstanceRef.current;
    const points = [];
    if (userLocation) points.push([userLocation.lat, userLocation.lon]);
    points.push([block.lat, block.lon]);
    if (destination) points.push([destination.lat, destination.lon]);
    if (points.length > 1) map.fitBounds(L.latLngBounds(points), { padding: [60, 60] });
  };

  // ── Search logic ──────────────────────────────────────────────
  const handleQueryChange = (val) => {
    setQuery(val);
    if (val.length === 0) { setDestination(null); setSuggestions([]); return; }
    if (val.length < 2) { setSuggestions([]); return; }
    const q = val.toLowerCase();
    const matches = SF_LOCATIONS.filter(loc =>
      loc.name.toLowerCase().includes(q) || loc.addr.toLowerCase().includes(q)
    ).slice(0, 6);
    setSuggestions(matches.map(loc => ({
      display_name: `${loc.name}, ${loc.addr}, San Francisco, CA`,
      lat: String(loc.lat), lon: String(loc.lon), _name: loc.name,
    })));
    setShowSuggestions(matches.length > 0);
  };

  const handleSelectSuggestion = (s) => {
    setDestination({ lat: parseFloat(s.lat), lon: parseFloat(s.lon), name: s.display_name.split(',')[0] });
    setQuery(s._name || s.display_name.split(',')[0]);
    setSuggestions([]);
    setShowSuggestions(false);
  };

  // ── Predict ───────────────────────────────────────────────────
  const handlePredict = async () => {
    if (!destination) { setError('Please enter a destination first'); return; }
    setLoading(true); setError(null); setSelectedBlock(null);
    const arrivalHour = showTimePicker ? (hour + Math.floor(minutesAway / 60)) % 24 : now.getHours();
    try {
      const res = await fetch('/api/predict/blocks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          lat: destination.lat, lon: destination.lon, radius_meters: 1500,
          hour: arrivalHour, day_of_week: conditions.day_of_week, month: conditions.month,
          is_raining: conditions.is_raining, event_intensity: conditions.event_intensity,
          is_holiday: conditions.is_holiday, is_school_day: conditions.is_school_day,
          temperature: conditions.temperature, minutes_away: showTimePicker ? minutesAway : 0,
        }),
      });
      if (!res.ok) throw new Error();
      const data = await res.json();
      setBlocks(data.blocks);
      if (data.blocks.length === 0) setError('No parking blocks found in this area. Try a different location.');
    } catch {
      setError('Could not get predictions. Make sure the API is running.');
    } finally {
      setLoading(false);
    }
  };

  // ── Block selection handlers ──────────────────────────────────
  const handleSelectBlock = (block) => {
    if (block) {
      setSelectedBlock(block);
      drawBlockRoutes(block);
    } else {
      setSelectedBlock(null);
      if (driveRouteRef.current) driveRouteRef.current.remove();
      if (walkRouteRef.current) walkRouteRef.current.remove();
      setRouteInfo(null);
      setDriveSteps([]);
    }
  };

  const handleNextBest = () => {
    const sorted = [...blocks].sort((a, b) => a.predicted_occupancy_pct - b.predicted_occupancy_pct);
    const nextIdx = (bestBlockIndex + 1) % sorted.length;
    setBestBlockIndex(nextIdx);
    const next = sorted[nextIdx];
    setSelectedBlock(next);
    drawBlockRoutes(next);
  };

  const handleBack = () => {
    setSelectedBlock(null);
    if (driveRouteRef.current) driveRouteRef.current.remove();
    if (walkRouteRef.current) walkRouteRef.current.remove();
    setRouteInfo(null);
    setDriveSteps([]);
  };

  // ── Render ────────────────────────────────────────────────────
  return (
    <div style={{ display: 'flex', height: '100vh', background: '#0d1b2a', color: 'white', fontFamily: '-apple-system, BlinkMacSystemFont, sans-serif', overflow: 'hidden' }}>

      {/* Sidebar */}
      <div style={{ width: 320, flexShrink: 0, display: 'flex', flexDirection: 'column', borderRight: '1px solid #1e3a52', background: '#0a1520' }}>

        {/* Header */}
        <div style={{ padding: '14px 16px', borderBottom: '1px solid #1e3a52', display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{ width: 30, height: 30, background: '#0d9488', borderRadius: 6, display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 800, fontSize: 15, flexShrink: 0 }}>P</div>
          <div style={{ flex: 1 }}>
            <div style={{ fontWeight: 700, fontSize: 15 }}>ParkCast SF</div>
            <div style={{ fontSize: 11, color: '#14b8a6' }}>Parking prediction engine</div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            <div style={{ width: 7, height: 7, borderRadius: '50%', background: apiStatus?.status === 'healthy' ? '#22c55e' : '#ef4444' }} />
            <span style={{ fontSize: 10, color: '#64748b' }}>{apiStatus?.status === 'healthy' ? 'Online' : 'Offline'}</span>
          </div>
        </div>

        <SearchBar
          query={query}
          onQueryChange={handleQueryChange}
          suggestions={suggestions}
          showSuggestions={showSuggestions}
          onShowSuggestions={setShowSuggestions}
          onSelect={handleSelectSuggestion}
          isRaining={conditions.is_raining === 1}
        />

        <TimePicker
          show={showTimePicker}
          onToggle={() => setShowTimePicker(!showTimePicker)}
          minutesAway={minutesAway}
          onMinutesChange={(val) => { setMinutesAway(val); setHour(now.getHours()); }}
          hour={hour}
        />

        {/* Find parking button */}
        <div style={{ padding: '12px 16px', borderBottom: '1px solid #1e3a52' }}>
          <button onClick={handlePredict} disabled={loading || !destination}
            style={{
              width: '100%', padding: '12px', borderRadius: 8, border: 'none',
              background: !destination ? '#0f2a3a' : loading ? '#0f6e56' : '#0d9488',
              color: !destination ? '#475569' : 'white',
              fontSize: 14, fontWeight: 700,
              cursor: (!destination || loading) ? 'not-allowed' : 'pointer',
              transition: 'background 0.2s',
            }}>
            {loading ? 'Predicting...' : !destination ? 'Enter a destination first' : showTimePicker ? `Find Parking (arriving in ${minutesAway} min)` : 'Find Parking Now'}
          </button>
          {error && (
            <div style={{ marginTop: 8, padding: '8px 12px', background: 'rgba(239,68,68,0.1)', border: '1px solid #7f1d1d', borderRadius: 6, color: '#fca5a5', fontSize: 12 }}>
              {error}
            </div>
          )}
        </div>

        {/* Results */}
        <div style={{ flex: 1, overflowY: 'auto' }}>
          {blocks.length === 0 && !loading && (
            <div style={{ padding: '24px 16px', textAlign: 'center', color: '#475569', fontSize: 13 }}>
              Search for a destination and tap Find Parking
            </div>
          )}
          <BlockList
            blocks={blocks}
            selectedBlock={selectedBlock}
            onSelectBlock={handleSelectBlock}
            routeInfo={routeInfo}
            driveSteps={driveSteps}
            onNextBest={handleNextBest}
            onBack={handleBack}
          />
        </div>
      </div>

      {/* Map */}
      <div style={{ flex: 1, position: 'relative' }}>
        <div ref={mapRef} style={{ width: '100%', height: '100%' }} />
        {loading && (
          <div style={{ position: 'absolute', inset: 0, background: 'rgba(10,21,32,0.6)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 999 }}>
            <div style={{ background: '#112031', border: '1px solid #1e3a52', borderRadius: 10, padding: '20px 32px', textAlign: 'center' }}>
              <div style={{ color: '#14b8a6', fontSize: 15, fontWeight: 700, marginBottom: 4 }}>Predicting parking...</div>
              <div style={{ color: '#64748b', fontSize: 12 }}>Analyzing blocks around your destination</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
