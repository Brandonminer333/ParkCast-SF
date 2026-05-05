/**
 * BlockList — parking block results with route info and turn-by-turn directions.
 *
 * Consolidates the results sidebar: route summary, "try next best" button,
 * block cards with demand indicators, and inline driving directions.
 */
'use client';

import ManeuverIcon from './ManeuverIcon';

const DEMAND_BG = {
  'Low': '#166534', 'Medium': '#854d0e',
  'High': '#9a3412', 'Very High': '#7f1d1d',
};

// Same red→amber→green gradient the map uses; fallback when the API
// row is missing `block.color`, otherwise the badge renders transparent
// and the basemap bleeds through.
const colorForPct = (pct) => {
  if (pct == null || Number.isNaN(pct)) return '#475569';
  const p = Math.max(0, Math.min(100, pct));
  const hue = (1 - p / 100) * 120;
  return `hsl(${hue.toFixed(0)}, 70%, 45%)`;
};

function RoutePanel({ routeInfo, onNextBest }) {
  return (
    <div style={{ padding: '10px 16px', background: '#0f2a3a', borderBottom: '1px solid #1e3a52' }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 8 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, background: '#112031', borderRadius: 6, padding: '8px 12px' }}>
          <div style={{ fontSize: 16 }}>🚗</div>
          <div style={{ flex: 1, fontSize: 12, color: '#94a3b8' }}>
            Drive <span style={{ color: '#cbd5e1' }}>your location</span> → <span style={{ color: '#cbd5e1' }}>parking</span>
          </div>
          <div style={{ fontSize: 14, fontWeight: 700, color: '#14b8a6' }}>
            {routeInfo.driveMin}<span style={{ fontSize: 10, fontWeight: 500, color: '#94a3b8' }}> min</span>
          </div>
        </div>
        {routeInfo.walkMin && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, background: '#112031', borderRadius: 6, padding: '8px 12px' }}>
            <div style={{ fontSize: 16 }}>🚶</div>
            <div style={{ flex: 1, fontSize: 12, color: '#94a3b8' }}>
              Walk <span style={{ color: '#cbd5e1' }}>parking</span> → <span style={{ color: '#cbd5e1' }}>destination</span>
            </div>
            <div style={{ fontSize: 14, fontWeight: 700, color: '#f8fafc' }}>
              {routeInfo.walkMin}<span style={{ fontSize: 10, fontWeight: 500, color: '#94a3b8' }}> min</span>
            </div>
          </div>
        )}
        {routeInfo.walkMin && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, background: '#0f1e2d', borderRadius: 6, padding: '8px 12px', border: '1px solid #1e3a52' }}>
            <div style={{ fontSize: 16 }}>⏱</div>
            <div style={{ flex: 1, fontSize: 12, color: '#94a3b8', fontWeight: 600 }}>Total trip</div>
            <div style={{ fontSize: 14, fontWeight: 700, color: '#f59e0b' }}>
              {routeInfo.driveMin + routeInfo.walkMin}<span style={{ fontSize: 10, fontWeight: 500, color: '#94a3b8' }}> min</span>
            </div>
          </div>
        )}
      </div>
      <button onClick={onNextBest} style={{ width: '100%', padding: '7px', borderRadius: 6, border: '1px solid #1e3a52', background: 'transparent', color: '#64748b', fontSize: 11, cursor: 'pointer' }}>
        Try next best option →
      </button>
    </div>
  );
}

function Directions({ driveSteps, routeInfo }) {
  if (!driveSteps.length) return null;
  return (
    <div style={{ marginBottom: 12, background: '#0a1422', borderRadius: 10, border: '1px solid #1e293b', overflow: 'hidden' }}>
      <div style={{ padding: '12px 14px', borderBottom: '1px solid #1e293b', display: 'flex', alignItems: 'baseline', gap: 8 }}>
        <div style={{ fontSize: 18, fontWeight: 700, color: '#f8fafc' }}>
          {routeInfo?.driveMin}<span style={{ fontSize: 12, fontWeight: 500, color: '#94a3b8' }}> min</span>
        </div>
        <div style={{ color: '#475569', fontSize: 11 }}>·</div>
        <div style={{ fontSize: 13, color: '#94a3b8' }}>
          {routeInfo?.driveMi != null ? `${routeInfo.driveMi.toFixed(1)} mi` : ''}
        </div>
        <div style={{ marginLeft: 'auto', fontSize: 10, fontWeight: 700, color: '#64748b', letterSpacing: 1.2 }}>DRIVE</div>
      </div>
      <div style={{ maxHeight: 240, overflowY: 'auto' }}>
        {driveSteps.map((s, i) => {
          const m = s.maneuver || {};
          const isArrive = m.type === 'arrive';
          const isDepart = m.type === 'depart';
          const street = s.name || s.ref || '';
          let verb, target;
          if (isDepart) { verb = 'Head'; target = street ? `onto ${street}` : ''; }
          else if (isArrive) { verb = 'Arrive at'; target = street || 'destination'; }
          else if (m.type === 'roundabout') { verb = 'Take the roundabout'; target = street ? `to ${street}` : ''; }
          else if (m.modifier) { verb = `Turn ${m.modifier}`; target = street ? `onto ${street}` : ''; }
          else { verb = 'Continue'; target = street ? `on ${street}` : ''; }
          const distMi = s.distance / 1609.344;
          const dist = distMi < 0.1 ? `${distMi.toFixed(2)} mi` : `${distMi.toFixed(1)} mi`;
          return (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '12px 14px', borderTop: i === 0 ? 'none' : '1px solid #122033' }}>
              <div style={{ width: 36, height: 36, borderRadius: 999, background: isArrive ? '#0d9488' : '#0f2031', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                <ManeuverIcon type={m.type} modifier={m.modifier} isArrive={isArrive} />
              </div>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: 13, lineHeight: 1.35, color: '#f1f5f9', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  <span style={{ fontWeight: 700 }}>{verb}</span>
                  {target && <span style={{ color: '#cbd5e1', fontWeight: 500 }}> {target}</span>}
                </div>
                {!isArrive && (
                  <div style={{ fontSize: 11.5, color: '#64748b', marginTop: 2 }}>{dist}</div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function BlockCard({ block, isSelected, onClick }) {
  const pct = block.predicted_occupancy_pct;
  const hasPct = pct != null && !Number.isNaN(pct);
  const badgeColor = block.color || colorForPct(pct);
  const street = block.street || 'Unnamed block';
  const demand = block.demand_level || '—';
  return (
    <div onClick={onClick}
      style={{
        padding: '10px 16px', borderBottom: '1px solid #0f1e2d', cursor: 'pointer',
        background: isSelected ? '#112031' : 'transparent',
      }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <div style={{
          width: 40, height: 40, borderRadius: 8, background: badgeColor,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontWeight: 800, fontSize: 13, flexShrink: 0, color: 'white',
        }}>
          {hasPct ? `${Math.round(pct)}%` : '—'}
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: 'white', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{street}</div>
          <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>
            {(block.distance_meters / 1609.344).toFixed(2)} mi away
          </div>
        </div>
        <div style={{ fontSize: 10, padding: '3px 7px', borderRadius: 4, background: DEMAND_BG[demand] || '#1e3a52', color: badgeColor, fontWeight: 600, flexShrink: 0 }}>
          {demand}
        </div>
      </div>
    </div>
  );
}

function BlockDetail({ block, driveSteps, routeInfo }) {
  const pct = block.predicted_occupancy_pct;
  return (
    <div style={{ marginTop: 8, padding: '8px 10px', background: '#0d1b2a', borderRadius: 6, fontSize: 12 }}>
      <div style={{ color: '#94a3b8', lineHeight: 1.5, marginBottom: 10 }}>
        {pct >= 85 && '🔴 Very hard to park — consider public transit or a nearby garage.'}
        {pct >= 70 && pct < 85 && '🟠 Limited spots — arrive early or check nearby blocks.'}
        {pct >= 40 && pct < 70 && '🟡 Decent chance of finding parking. Circle the block if needed.'}
        {pct < 40 && '🟢 Plenty of spaces — easy parking here!'}
      </div>
      <Directions driveSteps={driveSteps} routeInfo={routeInfo} />
    </div>
  );
}

export default function BlockList({
  blocks, selectedBlock, onSelectBlock, routeInfo,
  driveSteps, onNextBest, onBack,
}) {
  if (blocks.length === 0) return null;

  const isBlockSelected = (b) =>
    selectedBlock?.lat === b.lat && selectedBlock?.lon === b.lon;

  const visibleBlocks = selectedBlock
    ? blocks.filter(b => isBlockSelected(b))
    : blocks;

  return (
    <>
      {routeInfo && selectedBlock && (
        <RoutePanel routeInfo={routeInfo} onNextBest={onNextBest} />
      )}

      {selectedBlock ? (
        <button onClick={onBack}
          style={{
            display: 'flex', alignItems: 'center', gap: 8,
            width: '100%', padding: '10px 16px', background: '#0a1520',
            border: 'none', borderBottom: '1px solid #1e3a52',
            color: '#94a3b8', fontSize: 12, fontWeight: 600,
            cursor: 'pointer', textAlign: 'left',
          }}>
          ← Back to all {blocks.length} blocks
        </button>
      ) : (
        <div style={{ padding: '8px 16px', fontSize: 10, color: '#64748b', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', background: '#0a1520', borderBottom: '1px solid #1e3a52' }}>
          {blocks.length} blocks nearby · tap to see routes
        </div>
      )}

      {visibleBlocks.map(b => (
        <div key={`${b.lat},${b.lon}`}>
          <BlockCard
            block={b}
            isSelected={isBlockSelected(b)}
            onClick={() => onSelectBlock(isBlockSelected(b) ? null : b)}
          />
          {isBlockSelected(b) && (
            <div style={{ padding: '0 16px 10px' }}>
              <BlockDetail block={b} driveSteps={driveSteps} routeInfo={routeInfo} />
            </div>
          )}
        </div>
      ))}
    </>
  );
}
