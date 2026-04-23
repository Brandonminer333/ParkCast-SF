'use client';

import { useState, useEffect, useRef } from 'react';

export default function MapPage() {
  const mapRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const markersRef = useRef([]);
  const destMarkerRef = useRef(null);
  const userMarkerRef = useRef(null);
  const routeLayerRef = useRef(null);
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
  const [routeInfo, setRouteInfo] = useState(null);  // {driveMin, walkMin}
  const [bestBlockIndex, setBestBlockIndex] = useState(0);
  const [apiStatus, setApiStatus] = useState(null);
  const [showTimePicker, setShowTimePicker] = useState(false);
  const [hour, setHour] = useState(now.getHours());
  const [minutesAway, setMinutesAway] = useState(0);
  const [conditions, setConditions] = useState({
    is_raining: 0, has_nearby_event: 0,
    is_holiday: 0, is_school_day: 1,
    temperature: 62, month: now.getMonth() + 1,
    day_of_week: now.getDay() === 0 ? 6 : now.getDay() - 1,
  });

  // Auto-fetch weather conditions based on current time
  useEffect(() => {
    fetch('https://api.open-meteo.com/v1/forecast?latitude=37.7749&longitude=-122.4194&current=temperature_2m,precipitation,weathercode&temperature_unit=fahrenheit&timezone=America%2FLos_Angeles')
      .then(r => r.json())
      .then(data => {
        const current = data.current || {};
        setConditions(prev => ({
          ...prev,
          temperature:  current.temperature_2m || 62,
          is_raining:   (current.precipitation || 0) > 0.1 ? 1 : 0,
          bad_weather:  (current.weathercode || 0) >= 51 ? 1 : 0,
        }));
      })
      .catch(() => {}); // silently fail — use defaults
  }, []);

  // Get user's current location
  useEffect(() => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (pos) => setUserLocation({ lat: pos.coords.latitude, lon: pos.coords.longitude }),
        () => setUserLocation({ lat: 37.7749, lon: -122.4194 }) // default SF center
      );
    }
  }, []);

  // Load Leaflet
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

  // Init map
  useEffect(() => {
    if (!leafletLoaded || !mapRef.current || mapInstanceRef.current) return;
    const L = window.L;
    const map = L.map(mapRef.current).setView([37.7749, -122.4194], 14);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap contributors', maxZoom: 19,
    }).addTo(map);
    mapInstanceRef.current = map;
  }, [leafletLoaded]);

  // Show user location on map
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

  // Show destination marker
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

  // Draw routes when a block is selected
  const drawBlockRoutes = async (block) => {
    if (!mapInstanceRef.current || !leafletLoaded) return;
    const L = window.L;

    // Clear old routes
    if (driveRouteRef.current) driveRouteRef.current.remove();
    if (walkRouteRef.current) walkRouteRef.current.remove();
    setRouteInfo(null);

    // Drive route: user location → parking block
    if (userLocation) {
      try {
        const driveUrl = `https://router.project-osrm.org/route/v1/driving/${userLocation.lon},${userLocation.lat};${block.lon},${block.lat}?overview=full&geometries=geojson`;
        const driveRes = await fetch(driveUrl);
        const driveData = await driveRes.json();
        if (driveData.routes && driveData.routes[0]) {
          const coords = driveData.routes[0].geometry.coordinates.map(([lon, lat]) => [lat, lon]);
          driveRouteRef.current = L.polyline(coords, {
            color: '#0d9488', weight: 5, opacity: 0.85,
          }).addTo(mapInstanceRef.current);
          const driveSec = driveData.routes[0].duration;
          const driveMin = Math.ceil(driveSec / 60);

          // Walk route: parking block → destination
          let walkMin = null;
          if (destination) {
            try {
              const walkUrl = `https://router.project-osrm.org/route/v1/foot/${block.lon},${block.lat};${destination.lon},${destination.lat}?overview=full&geometries=geojson`;
              const walkRes = await fetch(walkUrl);
              const walkData = await walkRes.json();
              if (walkData.routes && walkData.routes[0]) {
                const walkCoords = walkData.routes[0].geometry.coordinates.map(([lon, lat]) => [lat, lon]);
                walkRouteRef.current = L.polyline(walkCoords, {
                  color: '#ffffff', weight: 3, opacity: 0.8, dashArray: '6 5',
                }).addTo(mapInstanceRef.current);
                walkMin = Math.ceil(walkData.routes[0].duration / 60);
              }
            } catch {}
          }
          setRouteInfo({ driveMin, walkMin });
        }
      } catch {}
    }

    // Fit map to show full journey
    const map = mapInstanceRef.current;
    const points = [];
    if (userLocation) points.push([userLocation.lat, userLocation.lon]);
    points.push([block.lat, block.lon]);
    if (destination) points.push([destination.lat, destination.lon]);
    if (points.length > 1) map.fitBounds(L.latLngBounds(points), { padding: [60, 60] });
  };

  // Clear routes when blocks list changes
  useEffect(() => {
    if (driveRouteRef.current) driveRouteRef.current.remove();
    if (walkRouteRef.current) walkRouteRef.current.remove();
    setRouteInfo(null);
    setSelectedBlock(null);
    setBestBlockIndex(0);
  }, [blocks]);

  // Render block markers
  useEffect(() => {
    if (!mapInstanceRef.current || !leafletLoaded) return;
    const L = window.L;
    markersRef.current.forEach(m => m.remove());
    markersRef.current = [];

    blocks.forEach(block => {
      const icon = L.divIcon({
        html: `<div style="
          background:${block.color};border:2px solid rgba(255,255,255,0.9);
          border-radius:6px;padding:3px 8px;font-size:11px;font-weight:700;
          color:white;white-space:nowrap;box-shadow:0 2px 6px rgba(0,0,0,0.35);cursor:pointer;
        ">${Math.round(block.predicted_occupancy_pct)}%</div>`,
        className: '', iconAnchor: [20, 12],
      });

      const marker = L.marker([block.lat, block.lon], { icon })
        .addTo(mapInstanceRef.current)
        .bindPopup(`
          <div style="font-family:sans-serif;min-width:210px;padding:4px">
            <b style="font-size:13px">${block.street}</b><br>
            <span style="color:${block.color};font-size:20px;font-weight:700">${block.predicted_occupancy_pct}%</span>
            <span style="color:#666;font-size:13px"> occupied</span><br>
            <span style="color:#333;font-size:13px">${block.demand_level} demand</span><br>
            <span style="color:#555;font-size:12px">~${block.available_spaces_estimate} of ${block.total_spaces} spaces free</span><br>
            <span style="color:#888;font-size:11px">${block.distance_meters}m from destination</span>
          </div>
        `);

      marker.on('click', () => setSelectedBlock(block));
      markersRef.current.push(marker);
    });
  }, [blocks, leafletLoaded]);

  // Check API health
  useEffect(() => {
    fetch('/api/health').then(r => r.json()).then(setApiStatus).catch(() => setApiStatus({ status: 'unavailable' }));
  }, []);

  // SF location database — no external API, works offline
  const SF_LOCATIONS = [
    {name:"USF Folger Building (101 Howard St)",addr:"101 Howard St",lat:37.7911,lon:-122.3926},
    {name:"University of San Francisco",addr:"2130 Fulton St",lat:37.7764,lon:-122.4511},
    {name:"SF State University",addr:"1600 Holloway Ave",lat:37.7246,lon:-122.4784},
    {name:"UC Hastings Law School",addr:"200 McAllister St",lat:37.7793,lon:-122.4177},
    {name:"Chase Center",addr:"1 Warriors Way",lat:37.7680,lon:-122.3877},
    {name:"Oracle Park (Giants)",addr:"24 Willie Mays Plaza",lat:37.7786,lon:-122.3893},
    {name:"Caltrain Station",addr:"700 4th St",lat:37.7764,lon:-122.3952},
    {name:"Embarcadero BART",addr:"Embarcadero",lat:37.7929,lon:-122.3966},
    {name:"Powell St BART",addr:"Powell St",lat:37.7844,lon:-122.4079},
    {name:"Civic Center BART",addr:"Civic Center",lat:37.7797,lon:-122.4143},
    {name:"16th St BART",addr:"16th St Mission",lat:37.7651,lon:-122.4197},
    {name:"24th St BART",addr:"24th St Mission",lat:37.7524,lon:-122.4184},
    {name:"UCSF Medical Center",addr:"505 Parnassus Ave",lat:37.7630,lon:-122.4578},
    {name:"UCSF Mission Bay",addr:"1975 4th St",lat:37.7631,lon:-122.3913},
    {name:"Zuckerberg SF General Hospital",addr:"1001 Potrero Ave",lat:37.7553,lon:-122.4058},
    {name:"St Marys Medical Center",addr:"450 Stanyan St",lat:37.7723,lon:-122.4545},
    {name:"Golden Gate Park",addr:"Golden Gate Park",lat:37.7694,lon:-122.4862},
    {name:"Dolores Park",addr:"Dolores Park",lat:37.7596,lon:-122.4269},
    {name:"Alamo Square (Painted Ladies)",addr:"Alamo Square",lat:37.7762,lon:-122.4344},
    {name:"Fishermans Wharf",addr:"Fishermans Wharf",lat:37.8080,lon:-122.4177},
    {name:"Pier 39",addr:"Pier 39",lat:37.8087,lon:-122.4098},
    {name:"Coit Tower",addr:"Telegraph Hill",lat:37.8024,lon:-122.4058},
    {name:"Moscone Center",addr:"747 Howard St",lat:37.7845,lon:-122.4004},
    {name:"SF City Hall",addr:"1 Dr Carlton B Goodlett Pl",lat:37.7793,lon:-122.4193},
    {name:"Ferry Building",addr:"1 Ferry Building",lat:37.7956,lon:-122.3935},
    {name:"Salesforce Tower",addr:"415 Mission St",lat:37.7895,lon:-122.3973},
    {name:"Transamerica Pyramid",addr:"600 Montgomery St",lat:37.7952,lon:-122.4028},
    {name:"Westfield SF Centre",addr:"865 Market St",lat:37.7841,lon:-122.4079},
    {name:"Castro District",addr:"Castro St & 18th St",lat:37.7609,lon:-122.4350},
    {name:"Mission District",addr:"16th St & Valencia St",lat:37.7645,lon:-122.4211},
    {name:"Haight-Ashbury",addr:"Haight St & Ashbury St",lat:37.7694,lon:-122.4469},
    {name:"North Beach",addr:"Columbus Ave & Broadway",lat:37.7990,lon:-122.4070},
    {name:"Chinatown",addr:"Grant Ave & California St",lat:37.7941,lon:-122.4078},
    {name:"Union Square",addr:"Union Square",lat:37.7879,lon:-122.4075},
    {name:"SoMa (South of Market)",addr:"Folsom St & 4th St",lat:37.7816,lon:-122.3975},
    {name:"Marina District",addr:"Chestnut St & Fillmore St",lat:37.8003,lon:-122.4360},
    {name:"Noe Valley",addr:"24th St & Noe St",lat:37.7502,lon:-122.4298},
    {name:"Richmond District",addr:"Clement St & 3rd Ave",lat:37.7830,lon:-122.4638},
    {name:"Sunset District",addr:"Irving St & 9th Ave",lat:37.7644,lon:-122.4658},
    {name:"Tenderloin",addr:"Turk St & Hyde St",lat:37.7836,lon:-122.4148},
    {name:"Hayes Valley",addr:"Hayes St & Octavia St",lat:37.7764,lon:-122.4238},
    {name:"Japantown",addr:"Post St & Buchanan St",lat:37.7852,lon:-122.4308},
    {name:"Dogpatch",addr:"3rd St & 22nd St",lat:37.7578,lon:-122.3888},
    {name:"Potrero Hill",addr:"18th St & Connecticut St",lat:37.7622,lon:-122.4038},
    {name:"Bernal Heights",addr:"Cortland Ave",lat:37.7396,lon:-122.4168},
    {name:"SFO Airport",addr:"San Francisco International Airport",lat:37.6213,lon:-122.3790},
    {name:"Fillmore Street",addr:"Fillmore St & Union St",lat:37.7984,lon:-122.4330},
    {name:"Howard Street",addr:"Howard St SoMa",lat:37.7820,lon:-122.3968},
    {name:"Market Street",addr:"Market St Downtown",lat:37.7879,lon:-122.4075},
    {name:"Valencia Street",addr:"Valencia St Mission",lat:37.7645,lon:-122.4211},
  ];

  const handleQueryChange = (val) => {
    setQuery(val);
    if (val.length === 0) { setDestination(null); setSuggestions([]); return; }
    if (val.length < 2) { setSuggestions([]); return; }
    const q = val.toLowerCase();
    const matches = SF_LOCATIONS.filter(loc =>
      loc.name.toLowerCase().includes(q) ||
      loc.addr.toLowerCase().includes(q)
    ).slice(0, 6);
    setSuggestions(matches.map(loc => ({
      display_name: `${loc.name}, ${loc.addr}, San Francisco, CA`,
      lat: String(loc.lat),
      lon: String(loc.lon),
      _name: loc.name,
    })));
    setShowSuggestions(matches.length > 0);
  };

  const handleSelectSuggestion = (s) => {
    setDestination({ lat: parseFloat(s.lat), lon: parseFloat(s.lon), name: s.display_name.split(',')[0] });
    setQuery(s._name || s.display_name.split(',')[0]);
    setSuggestions([]);
    setShowSuggestions(false);
  };

  const handlePredict = async () => {
    if (!destination) { setError('Please enter a destination first'); return; }
    setLoading(true); setError(null); setSelectedBlock(null);

    const arrivalHour = showTimePicker
      ? (hour + Math.floor(minutesAway / 60)) % 24
      : now.getHours();

    try {
      const res = await fetch('/api/predict/blocks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          lat: destination.lat,
          lon: destination.lon,
          radius_meters: 1500,
          hour: arrivalHour,
          day_of_week: conditions.day_of_week,
          month: conditions.month,
          is_raining: conditions.is_raining,
          has_nearby_event: conditions.has_nearby_event,
          is_holiday: conditions.is_holiday,
          is_school_day: conditions.is_school_day,
          temperature: conditions.temperature,
          minutes_away: showTimePicker ? minutesAway : 0,
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

  const demandBg = { 'Low': '#166534', 'Medium': '#854d0e', 'High': '#9a3412', 'Very High': '#7f1d1d' };

  return (
    <div style={{ display: 'flex', height: '100vh', background: '#0d1b2a', color: 'white', fontFamily: '-apple-system, BlinkMacSystemFont, sans-serif', overflow: 'hidden' }}>

      {/* ── Sidebar ── */}
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

        {/* Search */}
        <div style={{ padding: '14px 16px', borderBottom: '1px solid #1e3a52' }}>
          <div style={{ fontSize: 11, color: '#64748b', marginBottom: 8, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Where are you going?</div>
          <div style={{ position: 'relative' }}>
            <input
              value={query}
              onChange={e => handleQueryChange(e.target.value)}
              onFocus={() => suggestions.length > 0 && setShowSuggestions(true)}
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
                  <div key={i} onClick={() => handleSelectSuggestion(s)}
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

          {/* Auto conditions pills */}
          {conditions.is_raining === 1 && (
            <div style={{ marginTop: 8, display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              <span style={{ fontSize: 11, padding: '3px 8px', background: '#1e3a52', borderRadius: 4, color: '#94a3b8' }}>
                🌧 Raining detected
              </span>
            </div>
          )}
        </div>

        {/* Leave now / later toggle */}
        <div style={{ padding: '12px 16px', borderBottom: '1px solid #1e3a52' }}>
          <button onClick={() => setShowTimePicker(!showTimePicker)}
            style={{
              width: '100%', padding: '8px 14px', borderRadius: 7,
              border: '1px solid #1e3a52', background: showTimePicker ? '#112031' : 'transparent',
              color: showTimePicker ? '#14b8a6' : '#64748b', fontSize: 12, cursor: 'pointer',
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            }}>
            <span>{showTimePicker ? `Leaving in ${minutesAway} min (arriving ~${(hour + Math.floor(minutesAway/60)) % 24}:${String(minutesAway % 60).padStart(2,'0')})` : 'Leaving now'}</span>
            <span style={{ fontSize: 10 }}>{showTimePicker ? '▲' : '▼ Plan ahead'}</span>
          </button>

          {showTimePicker && (
            <div style={{ marginTop: 10, padding: '12px', background: '#112031', borderRadius: 8 }}>
              <div style={{ fontSize: 11, color: '#64748b', marginBottom: 6 }}>Leaving in: <b style={{ color: '#14b8a6' }}>{minutesAway} minutes</b></div>
              <input type="range" min={0} max={120} step={5} value={minutesAway}
                onChange={e => { setMinutesAway(+e.target.value); setHour(now.getHours()); }}
                style={{ width: '100%', accentColor: '#0d9488' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#475569', marginTop: 4 }}>
                <span>Now</span><span>30 min</span><span>1 hour</span><span>2 hours</span>
              </div>
            </div>
          )}
        </div>

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

          {blocks.length > 0 && (
            <>
              {routeInfo && selectedBlock && (
                <div style={{ padding: '10px 16px', background: '#0f2a3a', borderBottom: '1px solid #1e3a52' }}>
                  <div style={{ display: 'flex', gap: 10, marginBottom: 8 }}>
                    <div style={{ flex: 1, textAlign: 'center', background: '#112031', borderRadius: 6, padding: '6px 4px' }}>
                      <div style={{ fontSize: 18, fontWeight: 700, color: '#14b8a6' }}>{routeInfo.driveMin}m</div>
                      <div style={{ fontSize: 10, color: '#64748b' }}>🚗 drive</div>
                    </div>
                    {routeInfo.walkMin && <div style={{ flex: 1, textAlign: 'center', background: '#112031', borderRadius: 6, padding: '6px 4px' }}>
                      <div style={{ fontSize: 18, fontWeight: 700, color: '#f8fafc' }}>{routeInfo.walkMin}m</div>
                      <div style={{ fontSize: 10, color: '#64748b' }}>🚶 walk</div>
                    </div>}
                    {routeInfo.walkMin && <div style={{ flex: 1, textAlign: 'center', background: '#112031', borderRadius: 6, padding: '6px 4px' }}>
                      <div style={{ fontSize: 18, fontWeight: 700, color: '#f59e0b' }}>{routeInfo.driveMin + routeInfo.walkMin}m</div>
                      <div style={{ fontSize: 10, color: '#64748b' }}>⏱ total</div>
                    </div>}
                  </div>
                  <button onClick={() => {
                    const sorted = [...blocks].sort((a,b) => a.predicted_occupancy_pct - b.predicted_occupancy_pct);
                    const nextIdx = (bestBlockIndex + 1) % sorted.length;
                    setBestBlockIndex(nextIdx);
                    const next = sorted[nextIdx];
                    setSelectedBlock(next);
                    drawBlockRoutes(next);
                  }} style={{ width: '100%', padding: '7px', borderRadius: 6, border: '1px solid #1e3a52', background: 'transparent', color: '#64748b', fontSize: 11, cursor: 'pointer' }}>
                    Try next best option →
                  </button>
                </div>
              )}
              <div style={{ padding: '8px 16px', fontSize: 10, color: '#64748b', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', background: '#0a1520', borderBottom: '1px solid #1e3a52' }}>
                {blocks.length} blocks nearby · tap to see routes
              </div>
              {blocks.map(b => (
                <div key={b.block_id}
                  onClick={() => {
                    const isSelected = selectedBlock?.block_id === b.block_id;
                    setSelectedBlock(isSelected ? null : b);
                    if (!isSelected) drawBlockRoutes(b);
                    else {
                      if (driveRouteRef.current) driveRouteRef.current.remove();
                      if (walkRouteRef.current) walkRouteRef.current.remove();
                      setRouteInfo(null);
                    }
                  }}
                  style={{
                    padding: '10px 16px', borderBottom: '1px solid #0f1e2d', cursor: 'pointer',
                    background: selectedBlock?.block_id === b.block_id ? '#112031' : 'transparent',
                  }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <div style={{
                      width: 40, height: 40, borderRadius: 8, background: b.color,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      fontWeight: 800, fontSize: 13, flexShrink: 0,
                    }}>
                      {Math.round(b.predicted_occupancy_pct)}%
                    </div>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontSize: 12, fontWeight: 600, color: 'white', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{b.street}</div>
                      <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>
                        {b.available_spaces_estimate} spaces free · {b.distance_meters}m away
                      </div>
                    </div>
                    <div style={{ fontSize: 10, padding: '3px 7px', borderRadius: 4, background: demandBg[b.demand_level] || '#1e3a52', color: b.color, fontWeight: 600, flexShrink: 0 }}>
                      {b.demand_level}
                    </div>
                  </div>

                  {selectedBlock?.block_id === b.block_id && (
                    <div style={{ marginTop: 8, padding: '8px 10px', background: '#0d1b2a', borderRadius: 6, fontSize: 12 }}>
                      <div style={{ color: '#94a3b8', lineHeight: 1.5, marginBottom: 10 }}>
                        {b.predicted_occupancy_pct >= 85 && '🔴 Very hard to park — consider public transit or a nearby garage.'}
                        {b.predicted_occupancy_pct >= 70 && b.predicted_occupancy_pct < 85 && '🟠 Limited spots — arrive early or check nearby blocks.'}
                        {b.predicted_occupancy_pct >= 40 && b.predicted_occupancy_pct < 70 && '🟡 Decent chance of finding parking. Circle the block if needed.'}
                        {b.predicted_occupancy_pct < 40 && '🟢 Plenty of spaces — easy parking here!'}
                      </div>

                        <a href={`https://www.google.com/maps/dir/?api=1&destination=${b.lat},${b.lon}&travelmode=driving`}
                          target="_blank"
                          rel="noopener noreferrer"
                          style={{
                            display: 'block', width: '100%', padding: '9px',
                            background: '#0d9488', borderRadius: 7, border: 'none',
                            color: 'white', fontSize: 12, fontWeight: 700,
                            textAlign: 'center', textDecoration: 'none',
                            boxSizing: 'border-box',
                          }}>
                          Navigate to this parking block →
                        </a>
                      </div>
                    )}
                </div>
              ))}
            </>
          )}
        </div>
      </div>

      {/* ── Map ── */}
      <div style={{ flex: 1, position: 'relative' }}>
        <div ref={mapRef} style={{ width: '100%', height: '100%' }} />

        {/* Legend */}
        <div style={{
          position: 'absolute', bottom: 20, right: 16, background: 'rgba(10,21,32,0.92)',
          border: '1px solid #1e3a52', borderRadius: 8, padding: '10px 14px', zIndex: 1000,
        }}>
          <div style={{ fontSize: 10, color: '#64748b', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 6 }}>Occupancy</div>
          {[['#22c55e','< 40%  Easy'],['#f59e0b','40–70%  Moderate'],['#f97316','70–85%  Hard'],['#ef4444','> 85%  Very Hard']].map(([col, label]) => (
            <div key={col} style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 3 }}>
              <div style={{ width: 12, height: 12, borderRadius: 3, background: col, flexShrink: 0 }} />
              <span style={{ fontSize: 11, color: '#cbd5e1' }}>{label}</span>
            </div>
          ))}
          <div style={{ marginTop: 6, paddingTop: 6, borderTop: '1px solid #1e3a52' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 3 }}>
              <div style={{ width: 12, height: 12, borderRadius: '50%', background: '#3b82f6', flexShrink: 0 }} />
              <span style={{ fontSize: 11, color: '#cbd5e1' }}>Your location</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
              <div style={{ width: 12, height: 12, borderRadius: '50%', background: '#0d9488', flexShrink: 0 }} />
              <span style={{ fontSize: 11, color: '#cbd5e1' }}>Destination</span>
            </div>
          </div>
        </div>

        {/* Loading overlay */}
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
