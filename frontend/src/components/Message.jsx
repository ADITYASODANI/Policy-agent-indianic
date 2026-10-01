function formatReference(ref) {
  if (ref.label) return ref.label;
  return ref.section_number === "0"
    ? ref.section_title
    : `Section ${ref.section_number} — ${ref.section_title}`;
}

// Turns plain URLs into links; everything else stays as text.
// Trailing sentence punctuation ("...biz/.") is left outside the link.
function Linkify({ text }) {
  return text.split(/(https?:\/\/[^\s)]*[^\s).,;:!?'"’”])/g).map((part, i) =>
    /^https?:\/\//.test(part) ? (
      <a key={i} href={part} target="_blank" rel="noopener noreferrer">
        {part}
      </a>
    ) : (
      part
    )
  );
}

export default function Message({ message }) {
  if (message.role === "user") {
    return (
      <div className="msg-row user">
        <div className="bubble user-bubble">{message.text}</div>
      </div>
    );
  }

  const refs = message.references || [];
  return (
    <div className="msg-row assistant">
      <div className="avatar" aria-hidden="true">AI</div>
      <div className="bubble ai-bubble">
        {refs.length > 0 ? (
          <>
            <div className="label">Answer</div>
            <p>
              <Linkify text={message.text} />
            </p>
            <div className="label">Policy Reference</div>
            <ul className="refs">
              {refs.map((r) => (
                <li key={r.section_number}>{formatReference(r)}</li>
              ))}
            </ul>
            {message.recommendedAction && (
              <>
                <div className="label">Recommended Action</div>
                <p>
                  <Linkify text={message.recommendedAction} />
                </p>
              </>
            )}
          </>
        ) : (
          <p>
            <Linkify text={message.text} />
          </p>
        )}
      </div>
    </div>
  );
}
