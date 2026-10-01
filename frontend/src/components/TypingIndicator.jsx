export default function TypingIndicator() {
  return (
    <div className="msg-row assistant" aria-live="polite" aria-label="Assistant is typing">
      <div className="avatar" aria-hidden="true">AI</div>
      <div className="bubble ai-bubble typing">
        <span />
        <span />
        <span />
      </div>
    </div>
  );
}
