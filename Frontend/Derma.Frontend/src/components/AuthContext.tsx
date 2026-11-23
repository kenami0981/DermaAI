import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";

function getExpiry(token: string | null): number | null {
    if (!token) return null;
    try {
        const payload = JSON.parse(
            atob(token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/"))
        );
        return typeof payload.exp === "number" ? payload.exp * 1000 : null;
    } catch {
        return null;
    }
}

function readValidToken(): string | null {
    const token = localStorage.getItem("token");
    const exp = getExpiry(token);
    if (!token || exp === null || exp <= Date.now()) {
        localStorage.removeItem("token");
        return null;
    }
    return token;
}

interface AuthContextValue {
    isLoggedIn: boolean;
    login: (token: string) => void;
    logout: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
    const [token, setToken] = useState<string | null>(readValidToken);

    const login = useCallback((newToken: string) => {
        localStorage.setItem("token", newToken);
        setToken(newToken);
    }, []);

    const logout = useCallback(() => {
        localStorage.removeItem("token");
        setToken(null);
    }, []);

    // automatyczne wylogowanie w chwili wygaśnięcia tokenu
    useEffect(() => {
        const exp = getExpiry(token);
        if (!token || exp === null) return;
        const timeout = setTimeout(logout, Math.max(exp - Date.now(), 0));
        return () => clearTimeout(timeout);
    }, [token, logout]);

    // wylogowanie wywołane przez interceptor axiosa (401) oraz z innej karty
    useEffect(() => {
        const onLogout = () => setToken(null);
        const onStorage = (e: StorageEvent) => {
            if (e.key === "token") setToken(readValidToken());
        };
        window.addEventListener("auth-logout", onLogout);
        window.addEventListener("storage", onStorage);
        return () => {
            window.removeEventListener("auth-logout", onLogout);
            window.removeEventListener("storage", onStorage);
        };
    }, []);

    return (
        <AuthContext.Provider value={{ isLoggedIn: !!token, login, logout }}>
            {children}
        </AuthContext.Provider>
    );
}

export function useAuth() {
    const ctx = useContext(AuthContext);
    if (!ctx) throw new Error("useAuth musi być użyte wewnątrz AuthProvider");
    return ctx;
}