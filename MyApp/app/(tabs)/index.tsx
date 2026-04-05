import { useState, useEffect, useRef } from 'react';
import {
  StyleSheet, View, TextInput, TouchableOpacity,
  Text, ActivityIndicator, KeyboardAvoidingView, Platform, Alert
} from 'react-native';
import MapView, { Marker, Polyline, Region } from 'react-native-maps';
import * as Location from 'expo-location';

const GOOGLE_API_KEY = 'YOUR_KEY_HERE';

type Coordinate = { latitude: number; longitude: number };

function decodePolyline(encoded: string): Coordinate[] {
  const points: Coordinate[] = [];
  let index = 0, lat = 0, lng = 0;
  while (index < encoded.length) {
    let shift = 0, result = 0, b: number;
    do { b = encoded.charCodeAt(index++) - 63; result |= (b & 0x1f) << shift; shift += 5; } while (b >= 0x20);
    lat += result & 1 ? ~(result >> 1) : result >> 1;
    shift = 0; result = 0;
    do { b = encoded.charCodeAt(index++) - 63; result |= (b & 0x1f) << shift; shift += 5; } while (b >= 0x20);
    lng += result & 1 ? ~(result >> 1) : result >> 1;
    points.push({ latitude: lat / 1e5, longitude: lng / 1e5 });
  }
  return points;
}

export default function HomeScreen() {
  const mapRef = useRef<MapView>(null);
  const [userLocation, setUserLocation] = useState<Coordinate | null>(null);
  const [destination, setDestination] = useState('');
  const [destCoord, setDestCoord] = useState<Coordinate | null>(null);
  const [routeCoords, setRouteCoords] = useState<Coordinate[]>([]);
  const [loading, setLoading] = useState(false);
  const [duration, setDuration] = useState('');
  const [distance, setDistance] = useState('');

  useEffect(() => {
    (async () => {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission denied', 'Location access is needed for directions.');
        return;
      }
      const loc = await Location.getCurrentPositionAsync({});
      setUserLocation({ latitude: loc.coords.latitude, longitude: loc.coords.longitude });
    })();
  }, []);

  const getDirections = async () => {
    if (!userLocation || !destination.trim()) return;
    setLoading(true);
    setRouteCoords([]);
    setDestCoord(null);
    setDuration('');
    setDistance('');
    try {
      const origin = `${userLocation.latitude},${userLocation.longitude}`;
      const url = `https://maps.googleapis.com/maps/api/directions/json?origin=${origin}&destination=${encodeURIComponent(destination)}&key=${GOOGLE_API_KEY}`;
      const res = await fetch(url);
      const data = await res.json();
      console.log('Directions API response:', JSON.stringify(data, null, 2));
      if (data.status !== 'OK') {
        Alert.alert('Not found', 'Could not find directions to that location.');
        return;
      }
      const leg = data.routes[0].legs[0];
      setDuration(leg.duration.text);
      setDistance(leg.distance.text);
      const points = decodePolyline(data.routes[0].overview_polyline.points);
      setRouteCoords(points);
      const end = leg.end_location;
      setDestCoord({ latitude: end.lat, longitude: end.lng });

      // Fit map to show full route
      mapRef.current?.fitToCoordinates([userLocation, { latitude: end.lat, longitude: end.lng }], {
        edgePadding: { top: 80, right: 80, bottom: 80, left: 80 },
        animated: true,
      });
    } catch (e) {
      Alert.alert('Error', 'Something went wrong fetching directions.');
    } finally {
      setLoading(false);
    }
  };

  const initialRegion: Region | undefined = userLocation ? {
    latitude: userLocation.latitude,
    longitude: userLocation.longitude,
    latitudeDelta: 0.05,
    longitudeDelta: 0.05,
  } : undefined;

  return (
    <KeyboardAvoidingView style={styles.container} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
      <MapView ref={mapRef} style={styles.map} initialRegion={initialRegion} showsUserLocation>
        {destCoord && <Marker coordinate={destCoord} title="Destination" />}
        {routeCoords.length > 0 && (
          <Polyline coordinates={routeCoords} strokeColor="#4285F4" strokeWidth={4} />
        )}
      </MapView>

      <View style={styles.panel}>
        {duration ? (
          <View style={styles.infoRow}>
            <Text style={styles.infoText}>🕐 {duration}</Text>
            <Text style={styles.infoText}>📍 {distance}</Text>
          </View>
        ) : null}
        <View style={styles.inputRow}>
          <TextInput
            style={styles.input}
            placeholder="Enter destination..."
            placeholderTextColor="#999"
            value={destination}
            onChangeText={setDestination}
            onSubmitEditing={getDirections}
            returnKeyType="search"
          />
          <TouchableOpacity
            style={[styles.button, loading && styles.buttonDisabled]}
            onPress={getDirections}
            disabled={loading}
          >
            {loading
              ? <ActivityIndicator color="#fff" size="small" />
              : <Text style={styles.buttonText}>Go</Text>
            }
          </TouchableOpacity>
        </View>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  map: { flex: 1 },
  panel: {
    position: 'absolute',
    bottom: 40,
    left: 16,
    right: 16,
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 8,
    elevation: 5,
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 10,
  },
  infoText: { fontSize: 14, color: '#333', fontWeight: '600' },
  inputRow: { flexDirection: 'row', gap: 8 },
  input: {
    flex: 1,
    height: 44,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#ddd',
    paddingHorizontal: 12,
    fontSize: 15,
    color: '#333',
  },
  button: {
    backgroundColor: '#4285F4',
    borderRadius: 10,
    paddingHorizontal: 20,
    justifyContent: 'center',
    alignItems: 'center',
  },
  buttonDisabled: { opacity: 0.6 },
  buttonText: { color: '#fff', fontWeight: '700', fontSize: 15 },
});