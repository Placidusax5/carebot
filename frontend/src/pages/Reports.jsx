import { useEffect, useState } from "react";
import { Download, FileText } from "lucide-react";
import api, { historyApi, reportsApi } from "../services/api";
import RiskBadge from "../components/RiskBadge";

export default function Reports() {
  const [history, setHistory] = useState([]);
  const [reports, setReports] = useState([]);
  const [generating, setGenerating] = useState(null);

  const load = () => {
    historyApi.list().then((res) => setHistory(res.data));
    reportsApi.list().then((res) => setReports(res.data));
  };

  useEffect(load, []);

  const generate = async (predictionId) => {
    setGenerating(predictionId);
    try {
      await reportsApi.generate(predictionId);
      load();
    } finally {
      setGenerating(null);
    }
  };

  const download = async (reportId, title) => {
    const res = await api.get(`/reports/${reportId}/download`, { responseType: "blob" });
    const url = window.URL.createObjectURL(new Blob([res.data]));
    const a = document.createElement("a");
    a.href = url;
    a.download = `${title.replace(/[^a-z0-9]+/gi, "_")}.txt`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-10">
      <h1 className="text-2xl font-bold text-slate-900 mb-6">Reports</h1>

      <div className="grid md:grid-cols-2 gap-6">
        <div>
          <h2 className="font-semibold mb-3 text-slate-700">Generate from history</h2>
          <div className="card divide-y divide-slate-100">
            {history.length === 0 && <p className="text-sm text-slate-400 py-4">No predictions yet.</p>}
            {history.map((h) => (
              <div key={h.id} className="py-3 flex items-center justify-between gap-3">
                <div className="min-w-0">
                  <p className="font-medium text-sm truncate">{h.top_prediction}</p>
                  <p className="text-xs text-slate-400">{new Date(h.created_at).toLocaleDateString()}</p>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <RiskBadge level={h.risk_level} />
                  <button
                    onClick={() => generate(h.id)}
                    disabled={generating === h.id}
                    className="btn-secondary text-xs px-2.5 py-1.5"
                  >
                    {generating === h.id ? "Generating..." : "Generate Report"}
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h2 className="font-semibold mb-3 text-slate-700">Your Reports</h2>
          <div className="card divide-y divide-slate-100">
            {reports.length === 0 && <p className="text-sm text-slate-400 py-4">No reports generated yet.</p>}
            {reports.map((r) => (
              <div key={r.id} className="py-3 flex items-center justify-between gap-3">
                <div className="flex items-center gap-2 min-w-0">
                  <FileText className="w-4 h-4 text-slate-400 shrink-0" />
                  <div className="min-w-0">
                    <p className="font-medium text-sm truncate">{r.title}</p>
                    <p className="text-xs text-slate-400">{new Date(r.created_at).toLocaleDateString()}</p>
                  </div>
                </div>
                <button onClick={() => download(r.id, r.title)} className="btn-secondary text-xs px-2.5 py-1.5 flex items-center gap-1 shrink-0">
                  <Download className="w-3.5 h-3.5" /> Download
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
