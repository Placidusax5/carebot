import { useEffect, useRef, useState } from "react";
import { Send, Bot, User, Leaf, ShieldCheck } from "lucide-react";
import { chatApi } from "../services/api";
import RiskBadge from "../components/RiskBadge";

const QUICK_STARTS = [
  "I have a fever, sore throat and a runny nose",
  "Bad headache, nausea and light hurts my eyes",
  "My stomach hurts and I feel nauseous",
  "Dry cough, tiredness and loss of taste",
];

const prettifySymptom = (s) => s.replace(/_/g, " ");

export default function Assistant() {
  const [messages, setMessages] = useState([
    { role: "assistant", text: "Hi, I'm CareBot. Tell me what symptoms you're experiencing and I'll help you understand what might be going on." },
  ]);
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState(null);
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const send = async (rawText) => {
    const text = (rawText ?? input).trim();
    if (!text || loading) return;
    setMessages((prev) => [...prev, { role: "user", text }]);
    setInput("");
    setLoading(true);
    try {
      const res = await chatApi.send(text, sessionId);
      setSessionId(res.data.session_id);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: res.data.reply,
          prediction: res.data.prediction,
          extracted: res.data.extracted_symptoms || [],
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: "Sorry, something went wrong. Please try again." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  const isFresh = messages.length === 1;

  return (
    <div className="max-w-2xl mx-auto px-4 py-8 flex flex-col h-[85vh]">
      <h1 className="text-2xl font-bold text-slate-900 mb-1">AI Assistant</h1>
      <p className="text-slate-500 mb-4">Describe your symptoms in your own words.</p>

      <div className="flex-1 overflow-y-auto card space-y-4">
        {messages.map((m, i) => (
          <div key={i} className={`flex gap-2.5 ${m.role === "user" ? "justify-end" : ""}`}>
            {m.role === "assistant" && (
              <div className="w-8 h-8 rounded-full bg-brand-100 flex items-center justify-center shrink-0">
                <Bot className="w-4 h-4 text-brand-700" />
              </div>
            )}
            <div className={`max-w-[80%] ${m.role === "user" ? "order-first" : ""}`}>
              <div
                className={`rounded-2xl px-4 py-2.5 text-sm whitespace-pre-wrap ${
                  m.role === "user"
                    ? "bg-brand-600 text-white rounded-tr-sm"
                    : "bg-slate-100 text-slate-800 rounded-tl-sm"
                }`}
              >
                {m.text}
              </div>

              {/* Symptoms CareBot detected in the message */}
              {m.role === "assistant" && m.extracted?.length > 0 && (
                <div className="flex flex-wrap gap-1.5 mt-2">
                  {m.extracted.map((s) => (
                    <span
                      key={s}
                      className="text-xs bg-brand-50 text-brand-700 px-2 py-0.5 rounded-full capitalize"
                    >
                      {prettifySymptom(s)}
                    </span>
                  ))}
                </div>
              )}

              {m.prediction && (
                <div className="mt-2 border border-slate-200 rounded-xl p-3 bg-white">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-sm">{m.prediction.top_prediction}</span>
                    <RiskBadge level={m.prediction.risk_level} />
                  </div>

                  <ul className="text-xs text-slate-600 space-y-0.5 mb-2">
                    {m.prediction.predictions.map((p) => (
                      <li key={p.disease} className="flex items-center gap-2">
                        <span className="flex-1">{p.disease}</span>
                        <span className="text-slate-400">{p.confidence}%</span>
                      </li>
                    ))}
                  </ul>

                  {m.prediction.precautions?.length > 0 && (
                    <div className="mt-2 pt-2 border-t border-slate-100">
                      <h4 className="text-xs font-semibold text-slate-700 flex items-center gap-1 mb-1">
                        <ShieldCheck className="w-3.5 h-3.5" /> Precautions
                      </h4>
                      <ul className="text-xs text-slate-600 list-disc list-inside space-y-0.5">
                        {m.prediction.precautions.slice(0, 4).map((p) => (
                          <li key={p}>{p}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {m.prediction.home_remedies?.length > 0 && (
                    <div className="mt-2 rounded-lg border border-emerald-100 bg-emerald-50 p-2.5">
                      <h4 className="text-xs font-semibold text-emerald-800 flex items-center gap-1 mb-1">
                        <Leaf className="w-3.5 h-3.5" /> Home Remedies
                      </h4>
                      <ul className="text-xs text-emerald-900 list-disc list-inside space-y-0.5">
                        {m.prediction.home_remedies.slice(0, 4).map((r) => (
                          <li key={r}>{r}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </div>
            {m.role === "user" && (
              <div className="w-8 h-8 rounded-full bg-slate-200 flex items-center justify-center shrink-0">
                <User className="w-4 h-4 text-slate-600" />
              </div>
            )}
          </div>
        ))}

        {/* Quick-start suggestions for a fresh conversation */}
        {isFresh && !loading && (
          <div className="pl-10">
            <p className="text-xs text-slate-400 mb-2">Try one of these to get started:</p>
            <div className="flex flex-wrap gap-2">
              {QUICK_STARTS.map((q) => (
                <button
                  key={q}
                  onClick={() => send(q)}
                  className="text-xs text-left border border-slate-200 hover:border-brand-300 hover:bg-brand-50 text-slate-600 rounded-full px-3 py-1.5 transition-colors"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {loading && <div className="text-xs text-slate-400 pl-10">CareBot is typing...</div>}
        <div ref={bottomRef} />
      </div>

      <div className="flex gap-2 mt-4">
        <textarea
          rows={1}
          className="input-field resize-none"
          placeholder="Type your symptoms..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
        />
        <button onClick={() => send()} disabled={loading} className="btn-primary px-4">
          <Send className="w-4 h-4" />
        </button>
      </div>
      <p className="text-xs text-slate-400 mt-2">
        CareBot's chatbot uses built-in symptom-matching NLP (no external AI service required).
        Not a substitute for professional medical advice.
      </p>
    </div>
  );
}
