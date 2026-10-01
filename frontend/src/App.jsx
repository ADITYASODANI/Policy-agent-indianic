import { useEffect, useRef, useState } from "react";
import { clearHistory, fetchHistory, sendQuestion } from "./api.js";
import Header from "./components/Header.jsx";
import Message from "./components/Message.jsx";
import ChatInput from "./components/ChatInput.jsx";
import TypingIndicator from "./components/TypingIndicator.jsx";

const SESSION_KEY = "policy-assistant-session";
const SUGGESTIONS = [
  "Can I smoke inside the office?",
  "How do I report unethical behaviour?",
  "Kya main office laptop personal kaam ke liye use kar sakta hu?",
  "What happens if I accept a gift from a vendor?",
];

function newSessionId() {
  return crypto.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function getSessionId() {
  try {
    let id = localStorage.getItem(SESSION_KEY);
    if (!id) {
      id = newSessionId();
      localStorage.setItem(SESSION_KEY, id);
    }
    return id;
  } catch {
    return newSessionId();
  }
}

function toMessages(item) {
  return [
    { id: `${item.timestamp}-q`, role: "user", text: item.question },
    {
      id: `${item.timestamp}-a`,
      role: "assistant",
      text: item.answer,
      references: item.policy_references || [],
      recommendedAction: item.recommended_action,
    },
  ];
}

export default function App() {
  const [sessionId, setSessionId] = useState(getSessionId);
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const bottomRef = useRef(null);

  // Restore this browser's previous conversation; history is optional.
  useEffect(() => {
    fetchHistory(sessionId)
      .then((data) => setMessages(data.items.flatMap(toMessages)))
      .catch(() => {});
  }, [sessionId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function handleSend(question) {
    setError(null);
    setMessages((prev) => [...prev, { id: `${Date.now()}-q`, role: "user", text: question }]);
    setLoading(true);
    try {
      const data = await sendQuestion(question, sessionId);
      setMessages((prev) => [
        ...prev,
        {
          id: `${Date.now()}-a`,
          role: "assistant",
          text: data.answer,
          references: data.policy_references,
          recommendedAction: data.recommended_action,
        },
      ]);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleClear() {
    clearHistory(sessionId).catch(() => {});
    const id = newSessionId();
    try {
      localStorage.setItem(SESSION_KEY, id);
    } catch {
      /* storage unavailable: session lasts for this tab only */
    }
    setMessages([]);
    setError(null);
    setSessionId(id);
  }

  return (
    <div className="app">
      <Header onClear={handleClear} canClear={messages.length > 0 && !loading} />

      <main className="chat">
        {messages.length === 0 && !loading && (
          <div className="welcome">
            <h2>Hi! How can I help?</h2>
            <p>
              Ask me anything about the IndiaNIC <strong>Code of Conduct &amp; Ethics</strong>{" "}
              (Version 2, effective 01 January 2026).
            </p>
            <div className="suggestions">
              {SUGGESTIONS.map((s) => (
                <button key={s} className="suggestion" onClick={() => handleSend(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m) => (
          <Message key={m.id} message={m} />
        ))}
        {loading && <TypingIndicator />}
        {error && (
          <div className="error" role="alert">
            {error}
          </div>
        )}
        <div ref={bottomRef} />
      </main>

      <ChatInput onSend={handleSend} disabled={loading} />
    </div>
  );
}
