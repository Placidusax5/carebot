import { useState } from "react";
import { MapPin, Phone, Search } from "lucide-react";
import { hospitalsApi } from "../services/api";

export default function HospitalLocator() {
  const [city, setCity] = useState("");
  const [hospitals, setHospitals] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);

  const search = async (e) => {
    e.preventDefault();
    setLoading(true);
    setSearched(true);
    try {
      const res = await hospitalsApi.list(city);
      setHospitals(res.data);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto px-4 py-10">
      <h1 className="text-2xl font-bold text-slate-900 mb-1">Hospital Locator</h1>
      <p className="text-slate-500 mb-6">Find nearby hospitals and clinics.</p>

      <form onSubmit={search} className="flex gap-2 mb-6">
        <div className="relative flex-1">
          <Search className="w-4 h-4 absolute left-3 top-3 text-slate-400" />
          <input
            className="input-field pl-9"
            placeholder="Enter your city or area"
            value={city}
            onChange={(e) => setCity(e.target.value)}
          />
        </div>
        <button type="submit" disabled={loading} className="btn-primary">
          {loading ? "Searching..." : "Search"}
        </button>
      </form>

      <div className="space-y-3">
        {hospitals.map((h) => (
          <div key={h.name} className="card flex items-start justify-between gap-4">
            <div>
              <h3 className="font-semibold text-slate-900">{h.name}</h3>
              <p className="text-sm text-slate-500 flex items-center gap-1 mt-1">
                <MapPin className="w-3.5 h-3.5" /> {h.address} · {h.distance_km} km away
              </p>
              <p className="text-sm text-slate-500 flex items-center gap-1 mt-1">
                <Phone className="w-3.5 h-3.5" /> {h.phone}
              </p>
            </div>
            <a
              href={h.maps_link}
              target="_blank"
              rel="noreferrer"
              className="btn-secondary text-sm shrink-0"
            >
              Directions
            </a>
          </div>
        ))}
        {searched && !loading && hospitals.length === 0 && (
          <p className="text-slate-400 text-sm">No hospitals found.</p>
        )}
      </div>

      <p className="text-xs text-slate-400 mt-6">
        Results are illustrative sample data in this build. Connect a real geocoding/places
        provider (see backend/.env.example) for live results.
      </p>
    </div>
  );
}
