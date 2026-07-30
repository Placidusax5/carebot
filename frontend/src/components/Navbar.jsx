import { Link, useNavigate } from "react-router-dom";
import {
  Activity, Bot, Camera, Hospital, LayoutDashboard, FileText,
  Settings as SettingsIcon, LogOut, Stethoscope,
} from "lucide-react";
import { useAuth } from "../context/AuthContext";

const NAV_ITEMS = [
  { to: "/predictor", label: "Predictor", icon: Activity },
  { to: "/assistant", label: "AI Assistant", icon: Bot },
  { to: "/image-detection", label: "Image Detection", icon: Camera },
  { to: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { to: "/reports", label: "Reports", icon: FileText },
  { to: "/hospitals", label: "Hospitals", icon: Hospital },
];

export default function Navbar() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  return (
    <nav className="bg-white border-b border-slate-200 sticky top-0 z-10">
      <div className="max-w-7xl mx-auto px-4 flex items-center justify-between h-16">
        <Link to="/" className="flex items-center gap-2 font-bold text-brand-700 text-lg">
          <Stethoscope className="w-6 h-6" />
          CareBot AI
        </Link>

        {user && (
          <div className="hidden md:flex items-center gap-1">
            {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
              <Link
                key={to}
                to={to}
                className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm text-slate-600 hover:bg-brand-50 hover:text-brand-700 transition-colors"
              >
                <Icon className="w-4 h-4" /> {label}
              </Link>
            ))}
          </div>
        )}

        <div className="flex items-center gap-2">
          {user ? (
            <>
              <Link
                to="/settings"
                className="p-2 rounded-lg hover:bg-slate-100 text-slate-500"
                title="Settings"
              >
                <SettingsIcon className="w-5 h-5" />
              </Link>
              <button
                onClick={() => {
                  logout();
                  navigate("/login");
                }}
                className="flex items-center gap-1.5 text-sm text-slate-600 hover:text-red-600 px-3 py-2"
              >
                <LogOut className="w-4 h-4" /> Logout
              </button>
            </>
          ) : (
            <>
              <Link to="/login" className="btn-secondary text-sm">Login</Link>
              <Link to="/register" className="btn-primary text-sm">Sign Up</Link>
            </>
          )}
        </div>
      </div>
    </nav>
  );
}
