/**
 * ManeuverIcon — SVG turn-by-turn direction icons.
 *
 * Extracted from page.js where it was re-created on every render inside
 * an IIFE inside a map() inside JSX. Now a proper memoizable component.
 */
import { memo } from 'react';

function ManeuverIcon({ type, modifier, isArrive }) {
  const stroke = isArrive ? 'white' : '#14b8a6';
  const common = {
    width: 22, height: 22, viewBox: '0 0 24 24',
    fill: 'none', stroke, strokeWidth: 2.4,
    strokeLinecap: 'round', strokeLinejoin: 'round',
  };

  if (isArrive)
    return (<svg {...common}><circle cx="12" cy="10" r="4" fill="white"/><path d="M12 14v8" stroke="white"/></svg>);
  if (type === 'depart')
    return (<svg {...common}><path d="M12 20V6"/><path d="M5 13l7-7 7 7"/></svg>);
  if (type === 'roundabout')
    return (<svg {...common}><circle cx="12" cy="12" r="5"/><path d="M12 22v-5"/></svg>);

  switch (modifier) {
    case 'right':        return (<svg {...common}><path d="M5 19V11a4 4 0 0 1 4-4h10"/><path d="M14 2l5 5-5 5"/></svg>);
    case 'left':         return (<svg {...common}><path d="M19 19V11a4 4 0 0 0-4-4H5"/><path d="M10 2L5 7l5 5"/></svg>);
    case 'sharp right':  return (<svg {...common}><path d="M6 21V13a3 3 0 0 1 3-3h11"/><path d="M15 5l5 5-5 5"/></svg>);
    case 'sharp left':   return (<svg {...common}><path d="M18 21V13a3 3 0 0 0-3-3H4"/><path d="M9 5l-5 5 5 5"/></svg>);
    case 'slight right': return (<svg {...common}><path d="M7 22V11c0-3 2-5 5-5h6"/><path d="M14 2l5 4-5 4"/></svg>);
    case 'slight left':  return (<svg {...common}><path d="M17 22V11c0-3-2-5-5-5H6"/><path d="M10 2L5 6l5 4"/></svg>);
    case 'uturn':        return (<svg {...common}><path d="M5 21v-9a5 5 0 0 1 10 0v6"/><path d="M19 16l-4 4-4-4"/></svg>);
    case 'straight':     return (<svg {...common}><path d="M12 22V4"/><path d="M5 11l7-7 7 7"/></svg>);
    default:             return (<svg {...common}><circle cx="12" cy="12" r="2.5" fill={stroke}/></svg>);
  }
}

export default memo(ManeuverIcon);
