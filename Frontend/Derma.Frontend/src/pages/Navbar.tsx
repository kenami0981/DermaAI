
import { Link } from "react-router-dom";

export function Navbar() {
  return (
    <nav>
      <Link to="/">Strona główna</Link>
      <Link to="/login">Logowanie</Link>
      <Link to="/analysis">Analizy</Link>
      <Link to="/create-analysis">Dodaj analizę</Link>
    </nav>
  );
}
