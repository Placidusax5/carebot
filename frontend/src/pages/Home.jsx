import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { Activity, Bot, Camera, Hospital, LayoutDashboard, ShieldCheck } from "lucide-react";
import { useAuth } from "../context/AuthContext";

const FEATURES = [
  { icon: Activity, title: "Disease Prediction", desc: "Get the top 3 most probable conditions from your symptoms, with confidence scores." },
  { icon: Bot, title: "AI Assistant", desc: "Chat naturally about how you feel — CareBot extracts symptoms and asks smart follow-ups." },
  { icon: Camera, title: "Image Detection", desc: "Upload a photo for a preliminary skin condition assessment." },
  { icon: LayoutDashboard, title: "Personal Dashboard", desc: "Track your prediction history, risk trends, and download reports." },
  { icon: Hospital, title: "Hospital Locator", desc: "Find nearby hospitals and clinics with directions when you need care." },
  { icon: ShieldCheck, title: "Risk Assessment", desc: "Clear Low / Medium / High risk levels to help you gauge urgency." },
];

export default function Home() {
  const { user } = useAuth();

  return (
    <div>
      <section className="max-w-5xl mx-auto px-4 pt-20 pb-16 text-center">
        <motion.h1
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-4xl md:text-5xl font-bold text-slate-900 mb-4"
        >
          Meet <span className="text-brand-600">CareBot AI</span>
        </motion.h1>
        <p className="text-lg text-slate-600 max-w-2xl mx-auto mb-8">
          An intelligent healthcare assistant that helps you understand your symptoms,
          assess risk, and know when to see a doctor — powered by Machine Learning,
          Deep Learning, and NLP.
        </p>
        <div className="flex justify-center gap-3">
          <Link to={user ? "/predictor" : "/register"} className="btn-primary px-6 py-3 text-base">
            {user ? "Start a Prediction" : "Get Started Free"}
          </Link>
          <Link to={user ? "/assistant" : "/login"} className="btn-secondary px-6 py-3 text-base">
            {user ? "Talk to CareBot" : "Log In"}
          </Link>
        </div>
        <p className="text-xs text-slate-400 mt-6 max-w-lg mx-auto">
          CareBot AI provides preliminary, educational health information and does not
          replace professional medical advice, diagnosis, or treatment.
        </p>
      </section>

      <section className="max-w-6xl mx-auto px-4 pb-24 grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
        {FEATURES.map(({ icon: Icon, title, desc }) => (
          <div key={title} className="card">
            <div className="w-10 h-10 rounded-lg bg-brand-50 flex items-center justify-center mb-3">
              <Icon className="w-5 h-5 text-brand-600" />
            </div>
            <h3 className="font-semibold text-slate-900 mb-1">{title}</h3>
            <p className="text-sm text-slate-600">{desc}</p>
          </div>
        ))}
      </section>
    </div>
  );
}
