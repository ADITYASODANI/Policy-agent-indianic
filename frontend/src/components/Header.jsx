export default function Header({ onClear, canClear }) {
  return (
    <header className="header">
      <div className="brand">
        <div className="logo" aria-hidden="true">iN</div>
        <div>
          <h1>IndiaNIC Policy Assistant</h1>
          <p>Code of Conduct &amp; Ethics · Version 2</p>
        </div>
      </div>
      <button className="clear-btn" onClick={onClear} disabled={!canClear}>
        Clear chat
      </button>
    </header>
  );
}
