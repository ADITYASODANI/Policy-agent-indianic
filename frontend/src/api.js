const BASE_URL = import.meta.env.VITE_API_BASE_URL || "";
const UNREACHABLE = "Cannot reach the server. Please check that the backend is running.";

async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(`${BASE_URL}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch {
    throw new Error(UNREACHABLE);
  }

  const data = await res.json().catch(() => null);
  if (!res.ok) {
    const detail = typeof data?.detail === "string" ? data.detail : null;
    if (detail) throw new Error(detail);
    // A 5xx without our JSON body comes from the dev proxy: the API itself is down.
    if (res.status >= 500 && !data) throw new Error(UNREACHABLE);
    throw new Error(`Request failed (${res.status}). Please try again.`);
  }
  return data;
}

export const sendQuestion = (question, sessionId) =>
  request("/api/chat", {
    method: "POST",
    body: JSON.stringify({ question, session_id: sessionId }),
  });

export const fetchHistory = (sessionId) =>
  request(`/api/chat/history?session_id=${encodeURIComponent(sessionId)}`);

export const clearHistory = (sessionId) =>
  request(`/api/chat/history?session_id=${encodeURIComponent(sessionId)}`, { method: "DELETE" });
