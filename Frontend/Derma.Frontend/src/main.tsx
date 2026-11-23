import './index.css';
import { router } from './Routes';
import { RouterProvider } from 'react-router-dom';
import { StrictMode } from 'react';
import { AuthProvider } from './components/AuthContext';
import { createRoot } from 'react-dom/client';

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <AuthProvider>
      <RouterProvider router={router} />
    </AuthProvider>
  </StrictMode>
);

