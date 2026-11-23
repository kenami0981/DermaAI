import { createBrowserRouter, type RouteObject } from "react-router-dom";
import App from "./App";

import { MainPage } from "./pages/Main";
import { LoginPage } from "./pages/Login";
import { AnalysisList } from "./pages/AnalysisList";
import { AnalysisDetails } from "./pages/AnalysisDetails";
import { AnalysisCreate } from "./pages/AnalysisCreate";

import { ProtectedRoute } from "./components/ProtectedRoute";

export const routes: RouteObject[] = [
    {
        path: "/",
        element: <App />,
        children: [
            {
                path: "login",
                element: <LoginPage />
            },
            { path: "info", element: <MainPage /> },
            { index: true, element: <MainPage /> }, 
            {
                element: <ProtectedRoute />,
                children: [
                     
                    { path: "analysis", element: <AnalysisList /> },
                    { path: "analysis/:id", element: <AnalysisDetails /> },
                    { path: "create-analysis", element: <AnalysisCreate /> },
                ]
            }
        ]
    }
];

export const router = createBrowserRouter(routes);