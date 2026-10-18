import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import LandingPage from "./pages/LandingPage";
import ResultsPage from "./pages/ResultsPage";
import type { DiscoveryResponse } from "./api/types";

// The analysis result is passed from the landing page to the results page
// via router state. On a hard refresh the router state is lost, so we also
// stash the last result in sessionStorage and restore it as a fallback.
const STORAGE_KEY = "paperdna:lastResult";

export function stashResult(result: DiscoveryResponse): void {
  try {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(result));
  } catch {
    // Storage may be full or unavailable; the router state still works.
  }
}

export function readStashedResult(): DiscoveryResponse | null {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as DiscoveryResponse) : null;
  } catch {
    return null;
  }
}

export default function App() {
  const location = useLocation();
  // Prefer whatever the landing page put in router state.
  const result = (location.state as { result?: DiscoveryResponse } | null)
    ?.result;

  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route
        path="/results"
        element={<ResultsPage result={result ?? readStashedResult()} />}
      />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
