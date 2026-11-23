import { useEffect, useState } from "react";
import api from "../services/api";
import { type SkinAnalysis, AnalysisStatus, Tag } from "../dto/analysis";



export const AnalysisList = () => {
    const statusLabels: Record<AnalysisStatus, string> = {
    [AnalysisStatus.Pending]: "Oczekuje",
    [AnalysisStatus.Processing]: "W trakcie analizy",
    [AnalysisStatus.Completed]: "Zakończona",
    [AnalysisStatus.Failed]: "Błąd analizy",
};
const tagLabels: Record<Tag, string> = {
    [Tag.PoczatekCyklu]: "Początek cyklu",
    [Tag.SloneJedzenie]: "Słone jedzenie",
    [Tag.OstreJedzenie]: "Ostre jedzenie",
    [Tag.FastFood]: "Fast food",
    [Tag.Alkohol]: "Alkohol",
};
    const [analyses, setAnalyses] = useState<SkinAnalysis[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const load = async () => {
            try {
                const response = await api.get<SkinAnalysis[]>("/SkinAnalysis");
                setAnalyses(response.data);
            } catch (err: any) {
                setError(
                    err.response?.status === 401
                        ? "Sesja wygasła, zaloguj się ponownie."
                        : "Nie udało się pobrać analiz."
                );
            } finally {
                setLoading(false);
            }
        };
        load();
    }, []);

    if (loading) return <p>Ładowanie...</p>;
    if (error) return <p style={{ color: "crimson" }}>{error}</p>;
    if (analyses.length === 0) return <p>Nie masz jeszcze żadnych analiz.</p>;

    return (
        <div style={styles.grid}>
            {analyses.map((a) => (
                <div key={a.id} style={styles.card}>
                    <div style={styles.header}>
                        <strong>{new Date(a.uploadDate).toLocaleDateString("pl-PL")}</strong>
                        <span style={styles.badge}>{statusLabels[a.status]}</span>
                    </div>

                    {a.status === AnalysisStatus.Completed && a.result ? (
                        <div style={styles.result}>
                            <p>Wykryte zmiany: <strong>{a.result.acneCount}</strong></p>
                            <p>Wynik: <strong>{a.result.acneScore.toFixed(2)}</strong></p>
                        </div>
                    ) : (
                        <p style={styles.muted}>Wynik jeszcze niedostępny.</p>
                    )}

                    {a.notes && <p style={styles.notes}>{a.notes}</p>}

                    {a.tags.length > 0 && (
                        <div style={styles.tags}>
                            {a.tags.map((t) => (
                                <span key={t.id} style={styles.tag}>{tagLabels[t.tag]}</span>
                            ))}
                        </div>
                    )}
                </div>
            ))}
        </div>
    );
};

const styles: Record<string, React.CSSProperties> = {
    grid: {
        display: "grid",
        gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))",
        gap: "16px",
        padding: "16px",
    },
    card: {
        border: "1px solid #ddd",
        borderRadius: "12px",
        padding: "16px",
        background: "#fff",
        boxShadow: "0 2px 6px rgba(0,0,0,0.06)",
    },
    header: { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" },
    badge: { fontSize: "12px", padding: "2px 8px", borderRadius: "999px", background: "#eef" },
    result: { margin: "8px 0" },
    muted: { color: "#888" },
    notes: { fontStyle: "italic", color: "#555" },
    tags: { display: "flex", flexWrap: "wrap", gap: "6px", marginTop: "8px" },
    tag: { fontSize: "12px", padding: "2px 8px", borderRadius: "6px", background: "#f0f0f0" },
};