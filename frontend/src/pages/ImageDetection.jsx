import { useState } from "react";
import { UploadCloud, ImageOff, AlertTriangle } from "lucide-react";
import { imageApi } from "../services/api";

export default function ImageDetection() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFile = (f) => {
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError("");
  };

  const submit = async () => {
    if (!file) return;
    setLoading(true);
    setError("");
    try {
      const res = await imageApi.detect(file);
      setResult(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || "Could not analyze this image.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto px-4 py-10">
      <h1 className="text-2xl font-bold text-slate-900 mb-1">Image-Based Detection</h1>
      <p className="text-slate-500 mb-6">Upload a photo for a preliminary skin condition assessment.</p>

      <div className="card">
        <label
          className="border-2 border-dashed border-slate-300 rounded-xl flex flex-col items-center justify-center py-10 cursor-pointer hover:border-brand-400 transition-colors"
        >
          {preview ? (
            <img src={preview} alt="preview" className="max-h-56 rounded-lg mb-3" />
          ) : (
            <UploadCloud className="w-10 h-10 text-slate-400 mb-3" />
          )}
          <span className="text-sm text-slate-500">
            {file ? file.name : "Click to upload a JPG, PNG, or WEBP image"}
          </span>
          <input
            type="file"
            accept="image/jpeg,image/png,image/webp"
            className="hidden"
            onChange={(e) => handleFile(e.target.files[0])}
          />
        </label>

        <button onClick={submit} disabled={!file || loading} className="btn-primary w-full mt-5">
          {loading ? "Analyzing..." : "Analyze Image"}
        </button>
        {error && <p className="text-red-600 text-sm mt-3">{error}</p>}
      </div>

      {result && (
        <div className="card mt-6">
          {result.quality_warning && (
            <div className="flex items-start gap-2 mb-3 text-red-700 bg-red-50 border border-red-100 rounded-lg px-3 py-2 text-sm">
              <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
              <span>{result.quality_warning}</span>
            </div>
          )}

          <div className="flex justify-between items-center mb-3">
            <span className="font-semibold text-lg">{result.predicted_class}</span>
            <span className="text-slate-500 text-sm">{result.confidence}% confidence</span>
          </div>

          {result.predictions?.length > 0 && (
            <div className="space-y-2 mb-4">
              {result.predictions.map((p, i) => (
                <div key={p.label} className="flex items-center gap-3">
                  <span className="text-xs font-medium w-5 text-slate-400">#{i + 1}</span>
                  <div className="flex-1">
                    <div className="flex justify-between text-sm mb-1">
                      <span className={i === 0 ? "font-medium text-slate-800" : "text-slate-600"}>
                        {p.label}
                      </span>
                      <span className="text-slate-500">{p.confidence}%</span>
                    </div>
                    <div className="w-full bg-slate-100 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${i === 0 ? "bg-brand-500" : "bg-slate-300"}`}
                        style={{ width: `${Math.min(p.confidence, 100)}%` }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          <p className="text-sm text-slate-600 border-t border-slate-100 pt-3">
            {result.recommendation}
          </p>

          <div className="flex items-start gap-2 mt-3 text-amber-700 bg-amber-50 rounded-lg px-3 py-2 text-xs">
            <ImageOff className="w-4 h-4 shrink-0 mt-0.5" />
            {result.note}
          </div>
        </div>
      )}
    </div>
  );
}
