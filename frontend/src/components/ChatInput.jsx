import { useEffect, useRef, useState } from "react";

const MAX_LENGTH = 1000;

export default function ChatInput({ onSend, disabled }) {
  const [value, setValue] = useState("");
  const ref = useRef(null);

  useEffect(() => {
    if (!disabled) ref.current?.focus();
  }, [disabled]);

  function submit() {
    const question = value.trim();
    if (!question || disabled) return;
    onSend(question);
    setValue("");
  }

  function handleKeyDown(e) {
    // Enter sends, Shift+Enter adds a new line.
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  }

  return (
    <footer className="input-bar">
      <textarea
        ref={ref}
        rows={1}
        value={value}
        maxLength={MAX_LENGTH}
        placeholder="Ask about the company policy…"
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        aria-label="Your question"
      />
      <button className="send-btn" onClick={submit} disabled={disabled || !value.trim()}>
        Send
      </button>
    </footer>
  );
}
