import { useEffect, useState } from "react";
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from "recharts";
import { dashboardApi } from "../services/api";
import RiskBadge from "../components/RiskBadge";

const RISK_COLORS = { Low: "#22c55e", Medium: "#f59e0b", High: "#ef4444" };

export default function Dashboard() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    dashboardApi
      .get()
      .then((res) => setData(res.data))
      .catch(() => setError("Could not load dashboard data."));
  }, []);

  if (error) return <div className="max-w-5xl mx-auto px-4 py-10 text-red-600">{error}</div>;
  if (!data) return <div className="max-w-5xl mx-auto px-4 py-10 text-slate-500">Loading...</div>;

  const diseaseFreqData = Object.entries(data.disease_frequency).map(([name, count]) => ({ name, count }));
  const riskData = Object.entries(data.risk_breakdown).map(([name, value]) => ({ name, value }));
  const trendData = data.confidence_trend.map((t) => ({
    date: new Date(t.date).toLocaleDateString(),
    confidence: t.confidence,
  }));

  return (
    <div className="max-w-5xl mx-auto px-4 py-10">
      <h1 className="text-2xl font-bold text-slate-900 mb-6">Your Health Dashboard</h1>

      <div className="grid sm:grid-cols-3 gap-4 mb-6">
        <div className="card">
          <p className="text-sm text-slate-500">Total Predictions</p>
          <p className="text-3xl font-bold text-slate-900">{data.total_predictions}</p>
        </div>
        <div className="card">
          <p className="text-sm text-slate-500 mb-1">Risk Breakdown</p>
          <div className="flex gap-1.5 flex-wrap">
            {riskData.length === 0 && <span className="text-slate-400 text-sm">No data yet</span>}
            {riskData.map((r) => (
              <RiskBadge key={r.name} level={r.name} />
            ))}
          </div>
        </div>
        <div className="card">
          <p className="text-sm text-slate-500">Distinct Conditions Seen</p>
          <p className="text-3xl font-bold text-slate-900">{diseaseFreqData.length}</p>
        </div>
      </div>

      {data.total_predictions === 0 ? (
        <div className="card text-center text-slate-500 py-14">
          No predictions yet. Try the <span className="font-medium">Disease Predictor</span> or{" "}
          <span className="font-medium">AI Assistant</span> to get started.
        </div>
      ) : (
        <div className="grid md:grid-cols-2 gap-5">
          <div className="card">
            <h2 className="font-semibold mb-3">Disease Frequency</h2>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={diseaseFreqData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="name" tick={{ fontSize: 10 }} interval={0} angle={-30} textAnchor="end" height={60} />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="count" fill="#22a377" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="card">
            <h2 className="font-semibold mb-3">Risk Level Distribution</h2>
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie data={riskData} dataKey="value" nameKey="name" outerRadius={80} label>
                  {riskData.map((entry) => (
                    <Cell key={entry.name} fill={RISK_COLORS[entry.name] || "#94a3b8"} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div className="card md:col-span-2">
            <h2 className="font-semibold mb-3">Confidence Trend</h2>
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={trendData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="date" tick={{ fontSize: 10 }} />
                <YAxis domain={[0, 100]} />
                <Tooltip />
                <Line type="monotone" dataKey="confidence" stroke="#16825f" strokeWidth={2} dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="card md:col-span-2">
            <h2 className="font-semibold mb-3">Recent Predictions</h2>
            <div className="divide-y divide-slate-100">
              {data.recent_predictions.map((p) => (
                <div key={p.id} className="py-3 flex items-center justify-between">
                  <div>
                    <p className="font-medium text-sm">{p.top_prediction}</p>
                    <p className="text-xs text-slate-400">{new Date(p.created_at).toLocaleString()}</p>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-sm text-slate-500">{p.confidence}%</span>
                    <RiskBadge level={p.risk_level} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
