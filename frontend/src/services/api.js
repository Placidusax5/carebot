import axios from "axios";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const api = axios.create({ baseURL: API_URL });

api.interceptors.request.use((config) => {
  const token = localStorage.getItem("carebot_token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem("carebot_token");
      localStorage.removeItem("carebot_user");
      if (window.location.pathname !== "/login") {
        window.location.href = "/login";
      }
    }
    return Promise.reject(err);
  }
);

export const authApi = {
  register: (payload) => api.post("/auth/register", payload),
  login: (payload) => api.post("/auth/login", payload),
  me: () => api.get("/auth/me"),
};

export const predictApi = {
  symptoms: () => api.get("/symptoms"),
  predict: (symptoms) => api.post("/predict", { symptoms }),
};

export const chatApi = {
  send: (message, session_id) => api.post("/chat", { message, session_id }),
};

export const imageApi = {
  detect: (file) => {
    const form = new FormData();
    form.append("file", file);
    return api.post("/image-detect", form, {
      headers: { "Content-Type": "multipart/form-data" },
    });
  },
};

export const dashboardApi = {
  get: () => api.get("/dashboard"),
};

export const historyApi = {
  list: () => api.get("/history"),
};

export const reportsApi = {
  list: () => api.get("/reports"),
  generate: (predictionId) => api.post(`/reports/generate/${predictionId}`),
  downloadUrl: (reportId) => `${API_URL}/reports/${reportId}/download`,
};

export const hospitalsApi = {
  list: (city) => api.get("/hospitals", { params: { city } }),
  forDisease: (disease, city) =>
    api.get("/hospitals/for-disease", { params: { disease, city } }),
};

export default api;
