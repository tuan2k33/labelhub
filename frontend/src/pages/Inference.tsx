import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, flags, imageUrl, predictUrl, thumbUrl } from "../api";
import type { Annotation, Model, OverviewItem, PredictResult } from "../types";

const COLORS = ["#22c55e", "#f59e0b", "#3b82f6", "#ef4444", "#a855f7", "#14b8a6", "#ec4899", "#84cc16", "#f97316", "#06b6d4"];

interface Props {
  items: OverviewItem[];
  classes: string[];
  notify: (msg: string, error?: boolean) => void;
}
type Source = { kind: "file"; file: File; url: string } | { kind: "hash"; hash: string; url: string };

function Panel({ title, result, url, conf, classes, gt, error }: { title: string; result?: PredictResult; url: string; conf: number; classes: string[]; gt: Annotation[] | null; error?: string }) {
  const dets = useMemo(() => (result?.detections ?? []).filter((d) => d.confidence >= conf), [result, conf]);
  const counts = useMemo(() => {
    const c = new Map<string, number>();
    dets.forEach((d) => c.set(d.class_name, (c.get(d.class_name) ?? 0) + 1));
    return [...c.entries()];
  }, [dets]);
  const color = (name: string) => COLORS[Math.max(0, classes.indexOf(name)) % COLORS.length];
  const W = result?.width ?? 1, H = result?.height ?? 1;
  const fs = Math.max(10, W / 80);
  return (
    <section className="card panel">
      <div className="mhead">
        <b>{title}</b>
        {result && <span className="chip">{result.device === "cpu" ? "CPU" : "GPU"}</span>}
        {result && <span className="chip good" title="Whole request on the server, picture decoding included">{result.total_ms.toFixed(0)} ms</span>}
        {result && <span className="chip" title="Network forward pass only">model {result.inference_ms.toFixed(0)} ms</span>}
        {result?.cold && <span className="chip warn" title="The first run after loading includes warm-up and is left out of the latency stats">first run</span>}
      </div>
      <div className="stage">
        <div className="stagebox">
        <img src={url} alt="" />
        {result && (
          <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-label="Detections">
            {gt?.filter((a) => a.status === "accepted").map((a) => (
              <rect key={`g${a.id}`} x={(a.cx - a.w / 2) * W} y={(a.cy - a.h / 2) * H} width={a.w * W} height={a.h * H} fill="none" stroke="#fff" strokeWidth={2} strokeDasharray="6 4" vectorEffect="non-scaling-stroke" />
            ))}
            {dets.map((d, i) => {
              const b = d.box, label = `${d.class_name} ${d.confidence.toFixed(2)}`;
              const ty = b.y1 < fs * 1.5 ? b.y1 : b.y1 - fs * 1.4;
              return (
                <g key={i}>
                  <rect x={b.x1} y={b.y1} width={b.x2 - b.x1} height={b.y2 - b.y1} fill="none" stroke={color(d.class_name)} strokeWidth={2} vectorEffect="non-scaling-stroke" />
                  <rect x={b.x1} y={ty} width={label.length * fs * 0.58 + fs * 0.4} height={fs * 1.4} fill={color(d.class_name)} />
                  <text x={b.x1 + fs * 0.2} y={ty + fs * 1.05} fontSize={fs} fill="#0b0f19">{label}</text>
                </g>
              );
            })}
          </svg>
        )}
        </div>
      </div>
      {error ? <div className="notice">{error}</div> : (
        <div className="row wrap" style={{ marginTop: 8 }}>
          {result ? <span className="muted">{dets.length} box{dets.length === 1 ? "" : "es"} ≥ {conf.toFixed(2)}</span> : <span className="muted">Running…</span>}
          {counts.map(([n, c]) => <span key={n} className="chip"><span className="dot" style={{ background: color(n) }} />{n} {c}</span>)}
        </div>
      )}
    </section>
  );
}

export default function Inference({ items, classes, notify }: Props) {
  const [models, setModels] = useState<Model[]>([]);
  const [a, setA] = useState("");
  const [b, setB] = useState("");
  const [src, setSrc] = useState<Source | null>(null);
  const [conf, setConf] = useState(0.25);
  const [imgsz, setImgsz] = useState(0);
  const [res, setRes] = useState<Record<string, PredictResult | string>>({});
  const [showGt, setShowGt] = useState(false);
  const [gt, setGt] = useState<Annotation[] | null>(null);
  const [picking, setPicking] = useState(false);
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const runId = useRef(0);

  useEffect(() => {
    api.models().then((ms) => {
      const ok = ms.filter((m) => m.status === "done" && m.has_weights);
      setModels(ok);
      setA((cur) => cur || ok[0]?.name || "");
    }).catch((e) => notify(e.message, true));
  }, [notify]);

  const run = useCallback(async () => {
    if (!src || !a) return;
    const id = ++runId.current;
    setRes({});
    const names = [a, ...(b && b !== a ? [b] : [])];
    for (const n of names) {
      try {
        const r = await api.predict(n, src.kind === "file" ? { file: src.file } : { hash: src.hash }, 0.05, imgsz || undefined);
        if (id === runId.current) setRes((prev) => ({ ...prev, [n]: r }));
      } catch (e) {
        if (id === runId.current) setRes((prev) => ({ ...prev, [n]: (e as Error).message }));
      }
    }
    api.models().then((ms) => setModels(ms.filter((m) => m.status === "done" && m.has_weights))).catch(() => {});
  }, [src, a, b, imgsz]);
  useEffect(() => { run(); }, [run]);

  useEffect(() => {
    if (src?.kind === "hash" && showGt) api.annotations(src.hash).then(setGt).catch(() => setGt(null));
    else setGt(null);
  }, [src, showGt]);
  useEffect(() => () => { if (src?.kind === "file") URL.revokeObjectURL(src.url); }, [src]);

  const take = (file: File | undefined) => {
    if (!file) return;
    if (!file.type.startsWith("image/")) return notify("Choose an image file", true);
    setSrc({ kind: "file", file, url: URL.createObjectURL(file) });
  };
  const pick = (hash: string) => { setSrc({ kind: "hash", hash, url: imageUrl(hash) }); setPicking(false); };
  const random = () => { if (items.length) pick(items[Math.floor(Math.random() * items.length)].hash); };

  const model = (n: string) => models.find((m) => m.name === n);
  const label = (m: Model) => `${m.name}${m.metrics.map50 != null ? ` · mAP50 ${m.metrics.map50.toFixed(2)}` : ""}${m.inference?.p50_ms != null ? ` · ${m.inference.p50_ms.toFixed(0)} ms` : ""}`;
  const ra = res[a], rb = res[b];
  const asResult = (v?: PredictResult | string) => (typeof v === "object" ? v : undefined);
  const asError = (v?: PredictResult | string) => (typeof v === "string" ? v : undefined);
  const snippet = a
    ? `curl -X POST -F file=@photo.jpg -F conf=0.25 \\\n  ${predictUrl(a)}`
    : "";
  const py = a
    ? `import requests\nr = requests.post("${predictUrl(a)}",\n    files={"file": open("photo.jpg", "rb")}, data={"conf": 0.25})\nprint(r.json()["detections"])`
    : "";
  const copy = (text: string) => { navigator.clipboard?.writeText(text).then(() => notify("Copied")).catch(() => notify("Select the text and copy it", true)); };

  const cmpClasses = useMemo(() => {
    const ca = asResult(ra), cb = asResult(rb);
    if (!ca || !cb) return null;
    const count = (r: PredictResult) => { const m = new Map<string, number>(); r.detections.filter((d) => d.confidence >= conf).forEach((d) => m.set(d.class_name, (m.get(d.class_name) ?? 0) + 1)); return m; };
    const x = count(ca), y = count(cb);
    return [...new Set([...x.keys(), ...y.keys()])].sort().map((n) => [n, x.get(n) ?? 0, y.get(n) ?? 0] as const);
  }, [ra, rb, conf]);

  return (
    <div className="page">
      {models.length === 0 ? (
        <div className="empty">
          <h2>No trained model yet</h2>
          <p>Train one in Models or run Auto mode, then try it on a picture here.</p>
        </div>
      ) : (
        <>
          <div className="card" style={{ marginBottom: 16 }}>
            <div className="infergrid">
              <div className="field"><label htmlFor="model-a">Model</label>
                <select id="model-a" className="input" value={a} onChange={(e) => setA(e.target.value)}>{models.map((m) => <option key={m.name} value={m.name}>{label(m)}</option>)}</select></div>
              <div className="field"><label htmlFor="model-b">Compare with</label>
                <select id="model-b" className="input" value={b} onChange={(e) => setB(e.target.value)} disabled={models.length < 2}>
                  <option value="">{models.length < 2 ? "train another model to compare" : "none"}</option>
                  {models.filter((m) => m.name !== a).map((m) => <option key={m.name} value={m.name}>{label(m)}</option>)}
                </select></div>
              <div className="field"><label htmlFor="infer-imgsz">Image size</label>
                <select id="infer-imgsz" className="input" value={imgsz} onChange={(e) => setImgsz(Number(e.target.value))} disabled={flags.demo}>
                  <option value={0}>the model's own</option>
                  {[320, 416, 512, 640, 800, 1024].map((v) => <option key={v} value={v}>{v} px</option>)}
                </select></div>
              <div className="field slider"><label htmlFor="infer-conf">Confidence ≥ <b>{conf.toFixed(2)}</b></label>
                <input id="infer-conf" type="range" min={0.05} max={0.95} step={0.05} value={conf} onChange={(e) => setConf(Number(e.target.value))} /></div>
            </div>
            <div className="row wrap" style={{ marginTop: 12 }}>
              <input ref={fileInput} type="file" accept="image/*" hidden onChange={(e) => take(e.target.files?.[0])} />
              <button className="btn primary" onClick={() => fileInput.current?.click()}>Choose an image</button>
              <button className="btn" onClick={() => setPicking(true)} disabled={items.length === 0}>Pick from this dataset</button>
              <button className="btn" onClick={random} disabled={items.length === 0}>Random image</button>
              <label className="row muted" style={{ marginLeft: 8 }} title="Draws the boxes stored in the dataset as white dashed lines">
                <input id="show-gt" type="checkbox" checked={showGt} disabled={src?.kind !== "hash"} onChange={(e) => setShowGt(e.target.checked)} /> show dataset labels
              </label>
              {flags.demo && <span className="muted">Demo: pick one of the sample photos.</span>}
            </div>
          </div>

          {!src ? (
            <div
              className={`empty${dragging ? " dropping" : ""}`}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => { e.preventDefault(); setDragging(false); take(e.dataTransfer.files[0]); }}
            >
              <h2>Drop an image here</h2>
              <p>or choose one above. The model runs as soon as you do.</p>
            </div>
          ) : (
            <div
              className={`panels${b && b !== a ? " two" : ""}${dragging ? " dropping" : ""}`}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => { e.preventDefault(); setDragging(false); take(e.dataTransfer.files[0]); }}
            >
              <Panel title={a} result={asResult(ra)} error={asError(ra)} url={src.url} conf={conf} classes={classes} gt={gt} />
              {b && b !== a && <Panel title={b} result={asResult(rb)} error={asError(rb)} url={src.url} conf={conf} classes={classes} gt={gt} />}
            </div>
          )}

          {cmpClasses && (
            <section className="card" style={{ marginTop: 16 }}>
              <h3>Boxes per class</h3>
              <table className="cmp">
                <thead><tr><th>class</th><th>{a}</th><th>{b}</th></tr></thead>
                <tbody>{cmpClasses.map(([n, x, y]) => <tr key={n} className={x !== y ? "diff" : ""}><td>{n}</td><td>{x}</td><td>{y}</td></tr>)}</tbody>
              </table>
              {model(a)?.inference?.p50_ms != null && model(b)?.inference?.p50_ms != null && (
                <p className="muted" style={{ marginBottom: 0 }}>Typical latency: {a} {model(a)!.inference!.p50_ms!.toFixed(0)} ms, {b} {model(b)!.inference!.p50_ms!.toFixed(0)} ms (median of warm runs).</p>
              )}
            </section>
          )}

          <details className="card" style={{ marginTop: 16 }}>
            <summary>Use {a} from your own code</summary>
            <p className="muted">Send any image to the endpoint. The reply has the boxes, the latency and the device it ran on.</p>
            <pre className="log">{snippet}</pre>
            <div className="row"><button className="btn small" onClick={() => copy(snippet)}>Copy curl</button></div>
            <pre className="log">{py}</pre>
            <div className="row"><button className="btn small" onClick={() => copy(py)}>Copy Python</button></div>
          </details>
        </>
      )}

      {picking && (
        <div className="overlay" onMouseDown={() => setPicking(false)}>
          <div className="dialog wide" onMouseDown={(e) => e.stopPropagation()}>
            <h2>Pick an image</h2>
            <div className="grid pickgrid">
              {items.slice(0, 60).map((i) => (
                <button key={i.hash} className="tile" onClick={() => pick(i.hash)} title={i.filename}>
                  <img src={thumbUrl(i.hash, 160)} alt={i.filename} loading="lazy" />
                </button>
              ))}
            </div>
            {items.length > 60 && <p className="muted">Showing the first 60 of {items.length}. “Random image” draws from all of them.</p>}
            <div className="row" style={{ justifyContent: "flex-end", marginTop: 12 }}><button className="btn" onClick={() => setPicking(false)}>Close</button></div>
          </div>
        </div>
      )}
    </div>
  );
}
