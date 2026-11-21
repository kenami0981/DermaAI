
import axios from "axios";

const api = axios.create({
    baseURL: "https://localhost:7173/api",
    method:"POST",
    headers: {
        "Content-Type": "application/json"
    }
});

// Automatyczne dodawanie JWT do zapytań
api.interceptors.request.use((config) => {
    const token = localStorage.getItem("token");

    if (token) {
        config.headers.Authorization = `Bearer ${token}`;
    }

    return config;
});

export default api;
