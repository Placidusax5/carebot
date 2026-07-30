import { useEffect, useMemo, useState } from "react";
import { Search, X, Leaf, Stethoscope, MapPin, Phone, Building2, AlertTriangle } from "lucide-react";
import { predictApi, hospitalsApi } from "../services/api";
import RiskBadge from "../components/RiskBadge";

export default function Predictor() {
  const [allSymptoms, setAllSymptoms] = useState([]);
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState([]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Disease-specific hospital lookup
  const [city, setCity] = useState(() => localStorage.getItem("carebot_city") || "");
  const [hospitals, setHospitals] = useState([]);
  const [hospitalsLoading, setHospitalsLoading] = useState(false);
  const [hospitalsSearched, setHospitalsSearched] = useState(false);

  useEffect(() => {
    predictApi.symptoms().then((res) => setAllSymptoms(res.data.symptoms));
  }, []);

  const findHospitals = async () => {
    if (!result?.top_prediction) return;
    localStorage.setItem("carebot_city", city);
    setHospitalsLoading(true);
    setHospitalsSearched(true);
    try {
      const res = await hospitalsApi.forDisease(result.top_prediction, city || "your area");
      setHospitals(res.data);
    } catch {
      setHospitals([]);
    } finally {
      setHospitalsLoading(false);
    }
  };

  const filtered = useMemo(() => {
    if (!query) return [];
    const q = query.toLowerCase();
    return allSymptoms
      .filter((s) => s.includes(q.replace(/ /g, "_")) || s.replace(/_/g, " ").includes(q))
      .filter((s) => !selected.includes(s))
      .slice(0, 8);
  }, [query, allSymptoms, selected]);

  const addSymptom = (s) => {
    setSelected((prev) => [...prev, s]);
    setQuery("");
  };

  const removeSymptom = (s) => {
    setSelected((prev) => prev.filter((x) => x !== s));
  };

  const runPrediction = async () => {
    if (selected.length === 0) return;
    setError("");
    setLoading(true);
    setResult(null);
    setHospitals([]);
    setHospitalsSearched(false);
    try {
      const res = await predictApi.predict(selected);
      setResult(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || "Prediction failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto px-4 py-10">
      <h1 className="text-2xl font-bold text-slate-900 mb-1">Disease Predictor</h1>
      <p className="text-slate-500 mb-6">
        Select the symptoms you're experiencing to get the top 3 most probable conditions.
      </p>

      <div className="card mb-6">
        <label className="text-sm font-medium text-slate-700 block mb-2">Search symptoms</label>
        <div className="relative">
          <Search className="w-4 h-4 absolute left-3 top-3 text-slate-400" />
          <input
            className="input-field pl-9"
            placeholder="e.g. fever, headache, sore throat..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
        {filtered.length > 0 && (
          <div className="mt-2 border border-slate-200 rounded-lg divide-y max-h-56 overflow-y-auto">
            {filtered.map((s) => (
              <button
                key={s}
                onClick={() => addSymptom(s)}
                className="w-full text-left px-3 py-2 text-sm hover:bg-brand-50 capitalize"
              >
                {s.replace(/_/g, " ")}
              </button>
            ))}
          </div>
        )}

        {selected.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-4">
            {selected.map((s) => (
              <span
                key={s}
                className="flex items-center gap-1 bg-brand-50 text-brand-700 text-sm px-3 py-1 rounded-full capitalize"
              >
                {s.replace(/_/g, " ")}
                <button onClick={() => removeSymptom(s)}>
                  <X className="w-3.5 h-3.5" />
                </button>
              </span>
            ))}
          </div>
        )}

        <button
          onClick={runPrediction}
          disabled={selected.length === 0 || loading}
          className="btn-primary mt-5 w-full"
        >
          {loading ? "Analyzing..." : "Predict"}
        </button>
        {error && <p className="text-red-600 text-sm mt-3">{error}</p>}
      </div>

      {result && (
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold">Results</h2>
            <RiskBadge level={result.risk_level} />
          </div>

          <div className="space-y-3 mb-5">
            {result.predictions.map((p, i) => (
              <div key={p.disease} className="flex items-center gap-3">
                <span className="text-sm font-medium w-6 text-slate-400">#{i + 1}</span>
                <div className="flex-1">
                  <div className="flex justify-between text-sm mb-1">
                    <span className="font-medium text-slate-800">{p.disease}</span>
                    <span className="text-slate-500">{p.confidence}%</span>
                  </div>
                  <div className="w-full bg-slate-100 rounded-full h-2">
                    <div
                      className="bg-brand-500 h-2 rounded-full"
                      style={{ width: `${Math.min(p.confidence, 100)}%` }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="border-t border-slate-200 pt-4">
            <h3 className="font-medium text-slate-800 mb-1">About {result.top_prediction}</h3>
            <p className="text-sm text-slate-600 mb-3">{result.description}</p>
            <h3 className="font-medium text-slate-800 mb-1">Precautions</h3>
            <ul className="text-sm text-slate-600 list-disc list-inside space-y-0.5">
              {result.precautions.map((p) => (
                <li key={p}>{p}</li>
              ))}
            </ul>

            {result.home_remedies?.length > 0 && (
              <div className="mt-4 rounded-lg border border-emerald-100 bg-emerald-50 p-4">
                <h3 className="font-medium text-emerald-800 mb-1 flex items-center gap-1.5">
                  <Leaf className="w-4 h-4" />
                  Home Remedies
                </h3>
                <ul className="text-sm text-emerald-900 list-disc list-inside space-y-0.5">
                  {result.home_remedies.map((r) => (
                    <li key={r}>{r}</li>
                  ))}
                </ul>
                <p className="text-xs text-emerald-700/80 mt-2">
                  Supportive comfort measures only — not a treatment. Stop and seek care if
                  symptoms worsen.
                </p>
              </div>
            )}
          </div>

          {/* Where to get care — hospital matched to the predicted condition */}
          <div className="border-t border-slate-200 pt-4 mt-4">
            <h3 className="font-medium text-slate-800 mb-1 flex items-center gap-1.5">
              <Stethoscope className="w-4 h-4 text-brand-600" />
              Where to get care
            </h3>

            {result.risk_level === "High" && (
              <div className="flex items-start gap-2 text-sm text-red-700 bg-red-50 border border-red-100 rounded-lg p-2.5 mb-3">
                <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
                <span>
                  This assessment is high risk. If symptoms are severe or worsening, seek
                  emergency care right away.
                </span>
              </div>
            )}

            <p className="text-sm text-slate-600 mb-3">
              For <span className="font-medium">{result.top_prediction}</span>, the most relevant
              department is{" "}
              <span className="font-semibold text-brand-700">{result.recommended_specialty}</span>.
              Enter your city to find matching hospitals near you.
            </p>

            <div className="flex gap-2 mb-3">
              <div className="relative flex-1">
                <MapPin className="w-4 h-4 absolute left-3 top-3 text-slate-400" />
                <input
                  className="input-field pl-9"
                  placeholder="Your city or area"
                  value={city}
                  onChange={(e) => setCity(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && findHospitals()}
                />
              </div>
              <button
                onClick={findHospitals}
                disabled={hospitalsLoading}
                className="btn-primary shrink-0"
              >
                {hospitalsLoading ? "Finding..." : "Find hospitals"}
              </button>
            </div>

            <div className="space-y-2">
              {hospitals.map((h) => (
                <div
                  key={h.name}
                  className="flex items-start justify-between gap-3 border border-slate-200 rounded-lg p-3"
                >
                  <div>
                    <p className="font-medium text-sm text-slate-800 flex items-center gap-1.5">
                      <Building2 className="w-3.5 h-3.5 text-slate-400" />
                      {h.name}
                    </p>
                    {h.department && (
                      <span className="inline-block text-xs bg-brand-50 text-brand-700 px-2 py-0.5 rounded-full mt-1">
                        {h.department}
                      </span>
                    )}
                    <p className="text-xs text-slate-500 flex items-center gap-1 mt-1">
                      <MapPin className="w-3 h-3" /> {h.address} · {h.distance_km} km away
                    </p>
                    <p className="text-xs text-slate-500 flex items-center gap-1 mt-0.5">
                      <Phone className="w-3 h-3" /> {h.phone}
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
              {hospitalsSearched && !hospitalsLoading && hospitals.length === 0 && (
                <p className="text-sm text-slate-400">No hospitals found. Try a different city.</p>
              )}
            </div>
          </div>

          <p className="text-xs text-slate-400 mt-5 border-t border-slate-200 pt-3">
            {result.disclaimer} Hospital names are illustrative in this build; the Directions
            links open a live map search for the right kind of facility near you.
          </p>
        </div>
      )}
    </div>
  );
}
