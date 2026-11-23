import { NavLink } from "react-router-dom";
import "../styles/Navbar.css";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../components/AuthContext";

export function Navbar() {
  const navigate = useNavigate();
  const { isLoggedIn, logout } = useAuth();
    
     const handleAuthClick = () => {
    if (isLoggedIn) logout();
    navigate("/login");
  };
    
  return (
    <nav className="navbar">
      <NavLink to="/" className="navbar-brand">DermaAI</NavLink>

      <div className="navbar-links">
        <NavLink to="/" end>Strona główna</NavLink>
        <NavLink to="/info">Info</NavLink>
        {isLoggedIn && (
          <>
            <NavLink to="/analysis">Analizy</NavLink>
            <NavLink to="/create-analysis">Dodaj analizę</NavLink>
          </>
        )}
      </div>

      <button className="navbar-login" onClick={handleAuthClick}>
        {isLoggedIn ? "Wyloguj" : "Login"}
      </button>
    </nav>
  );
}