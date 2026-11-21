
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import api from "../services/api";
import "../styles/Login.css";

export function LoginPage() {
    const navigate = useNavigate();

    const [isRegister, setIsRegister] = useState(false);

    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [confirmPassword, setConfirmPassword] = useState("");

    const [error, setError] = useState("");
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        setError("");

        if (isRegister && password !== confirmPassword) {
            setError("Hasła nie są identyczne.");
            return;
        }

        setLoading(true);

        try {
            if (isRegister) {
                await api.post("/Account/register", {
                    email,
                    password
                });

                // Po rejestracji przechodzimy do logowania
                setIsRegister(false);
                setPassword("");
                setConfirmPassword("");
            } else {
                const response = await api.post("/Account/login", {
                    email,
                    password
                });

                const token = response.data.token;

                if (!token) {
                    throw new Error("Nie otrzymano tokenu JWT.");
                }

                localStorage.setItem("token", token);
                navigate("/", { replace: true });
            }
        } catch (err: any) {
            setError(
                err.response?.data?.message ||
                (isRegister
                    ? "Nie udało się utworzyć konta."
                    : "Nieprawidłowy email lub hasło.")
            );
        } finally {
            setLoading(false);
        }
    };

    const changeMode = (register: boolean) => {
        setIsRegister(register);
        setError("");
        setPassword("");
        setConfirmPassword("");
    };

    return (
        <div className="login-container">
            <div className="login-card">

                <h1>DermaAI</h1>
                <p className="login-subtitle">
                    Zadbaj o swoją skórę z pomocą AI
                </p>

                <div className="auth-tabs">
                    <button
                        type="button"
                        className={!isRegister ? "active" : ""}
                        onClick={() => changeMode(false)}
                    >
                        Logowanie
                    </button>

                    <button
                        type="button"
                        className={isRegister ? "active" : ""}
                        onClick={() => changeMode(true)}
                    >
                        Rejestracja
                    </button>
                </div>

                <form onSubmit={handleSubmit}>

                    <div className="form-group">
                        <label htmlFor="email">Email</label>
                        <input
                            id="email"
                            type="email"
                            placeholder="Wpisz swój email"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            required
                        />
                    </div>

                    <div className="form-group">
                        <label htmlFor="password">Hasło</label>
                        <input
                            id="password"
                            type="password"
                            placeholder="Wpisz hasło"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            required
                        />
                    </div>

                    {isRegister && (
                        <div className="form-group">
                            <label htmlFor="confirmPassword">
                                Powtórz hasło
                            </label>
                            <input
                                id="confirmPassword"
                                type="password"
                                placeholder="Powtórz hasło"
                                value={confirmPassword}
                                onChange={(e) =>
                                    setConfirmPassword(e.target.value)
                                }
                                required
                            />
                        </div>
                    )}

                    {error && (
                        <p className="login-error">{error}</p>
                    )}

                    <button
                        className="submit-button"
                        type="submit"
                        disabled={loading}
                    >
                        {loading
                            ? "Proszę czekać..."
                            : isRegister
                                ? "Utwórz konto"
                                : "Zaloguj się"}
                    </button>

                </form>

                <p className="auth-footer">
                    {isRegister
                        ? "Masz już konto? "
                        : "Nie masz jeszcze konta? "}

                    <span
                        onClick={() => changeMode(!isRegister)}
                    >
                        {isRegister ? "Zaloguj się" : "Zarejestruj się"}
                    </span>
                </p>

            </div>
        </div>
    );
}
