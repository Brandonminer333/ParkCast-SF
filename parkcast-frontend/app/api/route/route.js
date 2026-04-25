// Server-side proxy for Mapbox Directions so the access token stays out of
// the client bundle. Browser hits /api/route?profile=driving-traffic&coords=...
// — we call Mapbox here with the secret-ish MAPBOX_TOKEN env var.

const ALLOWED_PROFILES = new Set(["driving-traffic", "driving", "walking", "cycling"]);

export async function GET(request) {
  const { searchParams } = new URL(request.url);
  const profile = searchParams.get("profile") || "driving-traffic";
  const coords = searchParams.get("coords");

  if (!ALLOWED_PROFILES.has(profile)) {
    return Response.json({ error: `invalid profile: ${profile}` }, { status: 400 });
  }
  if (!coords || !/^[-\d.,;]+$/.test(coords)) {
    return Response.json({ error: "missing or malformed coords" }, { status: 400 });
  }

  const token = process.env.MAPBOX_TOKEN;
  if (!token) {
    return Response.json({ error: "MAPBOX_TOKEN not configured on server" }, { status: 500 });
  }

  const url =
    `https://api.mapbox.com/directions/v5/mapbox/${profile}/${coords}` +
    `?access_token=${token}&geometries=geojson&overview=full&steps=true`;

  try {
    const upstream = await fetch(url, { cache: "no-store" });
    const data = await upstream.json();
    return Response.json(data, { status: upstream.status });
  } catch (e) {
    return Response.json({ error: String(e) }, { status: 502 });
  }
}
