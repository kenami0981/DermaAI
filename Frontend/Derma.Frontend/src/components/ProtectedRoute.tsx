import { Navigate, Outlet } from "react-router-dom";
import { useAuth } from "../components/AuthContext";

export function ProtectedRoute() {
    const { isLoggedIn } = useAuth();
    return isLoggedIn ? <Outlet /> : <Navigate to="/login" replace />;
}