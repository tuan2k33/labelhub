import { useMemo, useRef, useState } from "react";
import { api, thumbUrl } from "../api";
import { useAutolabel } from "../useJob";
import type { OverviewItem } from "../types";

type Filter = "all" | "new" | "reviewed" | "gold";

interface Props {
  items: OverviewItem[];
  classes: string[];
  gpuBusy?: boolean;
  notify: (msg: string, error?: boolean) => void;
  onOpen: (hash: string, order: string[]) => void;
  onRefresh: () => void;
  onEditClasses: () => void;
}

const FILTERS: [Filter, string][] = [
  ["all", "All"],
  ["new", "Not opened yet"],
  ["reviewed", "Reviewed"],
  ["gold", "Gold"],
];

export default function Library({ items, classes, gpuBusy, notify, onOpen, onRefresh, onEditClasses }: Props) {
  const [filter, setFilter] = useState<Filter>("all");
  const [uploading, setUploading] = useState<{ done: number; total: number } | null>(null);
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const auto = useAutolabel(
    () => {
      notify("Auto-label finished");
      onRefresh();
    },
    (m) => notify(m, true),
  );

  const visible = useMemo(
    () =>
      items.filter((i) =>
        filter === "all" ? true : filter === "new" ? !i.reviewed : filter === "gold" ? i.gold : i.reviewed,
      ),
    [items, filter],
  );
  const fresh = items.filter((i) => !i.auto_done).length;

  const upload = async (files: File[]) => {
    const images = files.filter((f) => f.type.startsWith("image/"));
    if (!images.length) return notify("No image files found", true);
    setUploading({ done: 0, total: images.length });
    try {
      for (let i = 0; i < images.length; i += 20) {
        await api.upload(images.slice(i, i + 20));
        setUploading({ done: Math.min(i + 20, images.length), total: images.length });
      }
      notify(`Added ${images.length} image${images.length > 1 ? "s" : ""} (duplicates are skipped)`);
    } catch (e) {
      notify((e as Error).message, true);
    } finally {
      setUploading(null);
      onRefresh();
    }
  };

  const pct = auto.job && auto.job.total ? (auto.job.done / auto.job.total) * 100 : 0;

  return (
    <div
      className={`page${dragging ? " dropping" : ""}`}
      style={{ minHeight: "100%" }}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={(e) => e.currentTarget === e.target && setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        upload(Array.from(e.dataTransfer.files));
      }}
    >
      <div className="toolbar">
        <input ref={fileInput} type="file" accept="image/*" multiple hidden onChange={(e) => e.target.files && upload(Array.from(e.target.files))} />
        <button className="btn primary" onClick={() => fileInput.current?.click()} disabled={!!uploading}>
          Add images
        </button>
        <button
          className="btn"
          onClick={() => (classes.length ? auto.start(undefined, 0.5) : onEditClasses())}
          disabled={auto.running || !!gpuBusy || (classes.length > 0 && fresh === 0)}
          title={gpuBusy ? "GPU is busy training" : "SAM3 draws boxes (confidence 0.5 and up) for your classes. You fix them when you open each image."}
        >
          {classes.length ? `Auto-label ${fresh} new` : "Add classes to auto-label"}
        </button>
        {uploading && <span className="muted">Uploading {uploading.done}/{uploading.total}…</span>}
        {auto.job && (
          <span className="progress">
            <span className="muted">{auto.job.status === "loading model" ? "Loading SAM3…" : `${auto.job.done}/${auto.job.total}`}</span>
            <span className="bar"><div style={{ width: `${pct}%` }} /></span>
          </span>
        )}
        <span className="spacer" />
        <div className="filters" role="group" aria-label="Filter images">
          {FILTERS.map(([f, label]) => (
            <button key={f} className={filter === f ? "active" : ""} onClick={() => setFilter(f)}>
              {label}
            </button>
          ))}
        </div>
      </div>

      {items.length === 0 ? (
        <div className="empty">
          <h2>Drop images here</h2>
          <p>or use “Add images”. Then define your classes and let SAM3 draw the first boxes.</p>
        </div>
      ) : visible.length === 0 ? (
        <div className="empty">Nothing matches this filter.</div>
      ) : (
        <div className="grid">
          {visible.map((i) => (
            <button key={i.hash} className="tile" onClick={() => onOpen(i.hash, visible.map((v) => v.hash))} title={i.filename}>
              <img src={thumbUrl(i.hash)} alt={i.filename} loading="lazy" />
              <div className="meta">
                {i.reviewed ? <span className="chip good">✓ reviewed</span> : <span className="chip">new</span>}
                {i.gold && <span className="chip warn" title="Used only to score models">★ gold</span>}
                {i.accepted + i.proposed > 0 && <span className="chip">{i.accepted + i.proposed} box{i.accepted + i.proposed > 1 ? "es" : ""}</span>}
                {i.rejected > 0 && <span className="chip" title="Below the confidence threshold">{i.rejected} hidden</span>}
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
