import { useState } from "react";

interface Props {
  classes: string[];
  onSave: (names: string[]) => void;
  onClose: () => void;
}

export default function ClassesDialog({ classes, onSave, onClose }: Props) {
  const [names, setNames] = useState(classes);
  const [draft, setDraft] = useState("");

  const add = () => {
    const n = draft.trim();
    if (n && !names.includes(n)) setNames([...names, n]);
    setDraft("");
  };
  const move = (i: number, d: number) => {
    const next = [...names];
    [next[i], next[i + d]] = [next[i + d], next[i]];
    setNames(next);
  };

  return (
    <div className="overlay" onMouseDown={onClose}>
      <div className="dialog" onMouseDown={(e) => e.stopPropagation()}>
        <h2>Classes</h2>
        <p className="muted" style={{ marginTop: 0 }}>
          These are the text prompts SAM3 searches for, so use plain names ("person", "traffic cone"). Removing a class that
          existing boxes use will block the next label version until those boxes are fixed.
        </p>
        <div className="vlist">
          {names.length === 0 && <div className="muted">No classes yet.</div>}
          {names.map((n, i) => (
            <div key={n} className="vitem">
              <span className="row">
                <kbd>{i + 1}</kbd> {n}
              </span>
              <span className="row">
                <button className="btn small" disabled={i === 0} onClick={() => move(i, -1)} aria-label="Move up">↑</button>
                <button className="btn small" disabled={i === names.length - 1} onClick={() => move(i, 1)} aria-label="Move down">↓</button>
                <button className="btn small danger" onClick={() => setNames(names.filter((x) => x !== n))} aria-label={`Remove ${n}`}>✕</button>
              </span>
            </div>
          ))}
        </div>
        <div className="row">
          <input
            className="input"
            style={{ flex: 1 }}
            placeholder="Add a class and press Enter"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && add()}
            autoFocus
          />
          <button className="btn" onClick={add}>Add</button>
        </div>
        <div className="row" style={{ justifyContent: "flex-end", marginTop: 16 }}>
          <button className="btn" onClick={onClose}>Cancel</button>
          <button className="btn primary" onClick={() => onSave(names)}>Save</button>
        </div>
      </div>
    </div>
  );
}
