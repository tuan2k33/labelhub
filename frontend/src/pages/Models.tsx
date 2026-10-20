import { useCallback, useEffect, useMemo, useState } from "react";
import { api, flags, weightsUrl } from "../api";
import type { ImageVersion, LabelVersion, Model, ServerStatus, TrainPreview } from "../types";

interface Props {
  status: ServerStatus | null;
  notify: (msg: string, error?: boolean) => void;
  onStatusChange: () => void;
}

const fmt = (iso: string) => new Date(iso).toLocaleString();
const f3 = (n?: number) => (n == null ? "—" : n.toFixed(3));
const autoShare = (src: Record<string, number>) => {
  const total = Object.values(src).reduce((a, b) => a + b, 0);
  const auto = Object.entries(src).filter(([k]) => k.startsWith("auto:")).reduce((a, [, v]) => a + v, 0);
  return total ? auto / total : 0;
};
const sourcesText = (src: Record<string, number>) => {
  const total = Object.values(src).reduce((a, b) => a + b, 0);
  if (!total) return "label sources unknown";
  return Object.entries(src).map(([k, v]) => `${Math.round((v / total) * 100)}% ${k === "human" ? "human" : k.replace("auto:", "")}`).join(", ");
};
const TONE: Record<Model["status"], string> = { running: "warn", done: "good", failed: "bad", cancelled: "" };

function ModelCard({ m, onTrainFrom, notify, onChanged }: { m: Model; onTrainFrom: (name: string) => void; notify: Props["notify"]; onChanged: () => void }) {
  const [log, setLog] = useState<string | null>(null);
  const live = m.metrics.live;
  const pct = m.epochs ? Math.min(100, (m.epoch / m.epochs) * 100) : 0;
  const toggleLog = async () => {
    if (log !== null) return setLog(null);
    try { setLog((await api.modelLog(m.name)).log || "(empty)"); } catch (e) { notify((e as Error).message, true); }
  };
  const cancel = async () => {
    try { await api.cancelModel(m.name); notify(`Cancelled ${m.name}`); onChanged(); } catch (e) { notify((e as Error).message, true); }
  };
  const mm = m.metrics;
  return (
    <div className="card">
      <div className="mhead">
        <b>{m.name}</b>
        <span className={`chip ${TONE[m.status]}`}>{m.status}</span>
        {m.imported && <span className="chip">imported</span>}
        {m.parent && <span className="chip">fine-tuned from {m.parent}</span>}
        <span className="spacer" />
        <span className="muted">{fmt(m.created_at)}</span>
      </div>
      {m.note && <div>{m.note}</div>}
      <div className="lineage">
        images <b>{m.image_version}</b> → labels <b>{m.label_version}</b> · {m.imported ? "trained elsewhere" : `${m.base}, ${m.config.epochs} epochs, ${m.config.imgsz}px`}
      </div>

      {m.status === "running" && (
        <>
          <div className="progress">
            <span className="muted">{m.epoch === 0 ? "preparing data…" : `epoch ${m.epoch}/${m.epochs}`}</span>
            <span className="bar"><div style={{ width: `${pct}%` }} /></span>
            {live?.mAP50 != null && <span className="chip">mAP50 {f3(live.mAP50)}</span>}
          </div>
        </>
      )}
      {m.error && <div className="notice">{m.error}</div>}
      {m.inference?.p50_ms != null && (
        <div className="muted" style={{ fontSize: 12 }}>
          Inference: median {m.inference.p50_ms.toFixed(0)} ms, 95th percentile {m.inference.p95_ms?.toFixed(0)} ms over {m.inference.warm_runs} runs ({m.inference.device === "cpu" ? "CPU" : "GPU"})
        </div>
      )}

      {m.status === "done" && (
        <>
          <div className="mgrid">
            <div className="metric"><div className="v">{f3(mm.map50)}</div><div className="l">mAP50</div></div>
            <div className="metric"><div className="v">{f3(mm.map50_95)}</div><div className="l">mAP50-95</div></div>
            <div className="metric"><div className="v">{f3(mm.precision)}</div><div className="l">precision</div></div>
            <div className="metric"><div className="v">{f3(mm.recall)}</div><div className="l">recall</div></div>
          </div>
          {mm.gold && (
            <div className="goldrow">
              <span className="chip warn">★ gold</span>
              <span><b>{f3(mm.gold.map50)}</b> mAP50 on {mm.gold.images} human-checked image{mm.gold.images > 1 ? "s" : ""} (never trained on)</span>
            </div>
          )}
          {!mm.gold && <div className="muted" style={{ fontSize: 12 }}>No gold images, so the scores above only show how well the model matches its labels.</div>}
          {mm.per_class_map50 && (
            <div className="kbars">
              {Object.entries(mm.per_class_map50).map(([c, v]) => (
                <span key={c} style={{ display: "contents" }}>
                  <span>{c}</span>
                  <span className="kbar"><div style={{ width: `${v * 100}%` }} /></span>
                  <span className="muted">{v.toFixed(2)}</span>
                </span>
              ))}
            </div>
          )}
        </>
      )}

      <div className="row wrap" style={{ marginTop: 12 }}>
        {m.status === "running" && <button className="btn small danger" onClick={cancel}>Cancel</button>}
        {m.has_weights && (flags.demo
          ? <button className="btn small" disabled title="Weights stay on the machine that trained them">Download weights</button>
          : <a className="btn small" href={weightsUrl(m.name)} download>Download weights</a>)}
        {m.status === "done" && m.has_weights && <button className="btn small" onClick={() => onTrainFrom(m.name)}>Fine-tune…</button>}
        {!m.imported && <button className="btn small" onClick={toggleLog}>{log === null ? "Log" : "Hide log"}</button>}
      </div>
      {log !== null && <pre className="log">{log}</pre>}
    </div>
  );
}

export default function Models({ status, notify, onStatusChange }: Props) {
  const [models, setModels] = useState<Model[]>([]);
  const [iv, setIv] = useState<ImageVersion[]>([]);
  const [lv, setLv] = useState<LabelVersion[]>([]);
  const [form, setForm] = useState({ image_version: "", label_version: "", base: "yolo26n.pt", epochs: 50, imgsz: 640, batch: 8, note: "", parent: "" });
  const [preview, setPreview] = useState<TrainPreview | null>(null);
  const [previewErr, setPreviewErr] = useState("");
  const [busy, setBusy] = useState(false);
  const [unopened, setUnopened] = useState(false);
  const [imp, setImp] = useState({ image_version: "", label_version: "", map50: "", note: "" });
  const [impFile, setImpFile] = useState<File | null>(null);

  const load = useCallback(async () => {
    try { setModels(await api.models()); } catch (e) { notify((e as Error).message, true); }
  }, [notify]);

  useEffect(() => {
    load();
    Promise.all([api.imageVersions(), api.labelVersions()]).then(([a, b]) => {
      setIv(a);
      setLv(b);
      setForm((f) => ({ ...f, image_version: f.image_version || a[a.length - 1]?.name || "" }));
      setImp((f) => ({ ...f, image_version: f.image_version || a[a.length - 1]?.name || "", label_version: f.label_version || b[b.length - 1]?.name || "" }));
    }).catch((e) => notify(e.message, true));
  }, [load, notify]);

  const anyRunning = models.some((m) => m.status === "running");
  useEffect(() => {
    if (!anyRunning) return;
    const t = window.setInterval(() => { load(); onStatusChange(); }, 3000);
    return () => window.clearInterval(t);
  }, [anyRunning, load, onStatusChange]);

  useEffect(() => {
    if (!form.image_version) return setPreview(null);
    api.trainPreview(form.image_version, form.label_version || undefined, unopened)
      .then((p) => { setPreview(p); setPreviewErr(""); })
      .catch((e) => { setPreview(null); setPreviewErr(e.message); });
  }, [form.image_version, form.label_version, lv.length, unopened]);

  const set = (k: keyof typeof form) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) =>
    setForm({ ...form, [k]: ["epochs", "imgsz", "batch"].includes(k) ? Number(e.target.value) || 0 : e.target.value });

  const gpuBusy = status?.gpu_busy;
  const share = preview ? autoShare(preview.label_sources) : 0;
  const doneModels = useMemo(() => models.filter((m) => m.status === "done" && m.has_weights), [models]);

  const start = async () => {
    setBusy(true);
    try {
      const m = await api.train({ ...form, label_version: form.label_version || null, parent: form.parent || null, include_unopened: unopened });
      notify(`Training ${m.name} started`);
      await load();
      onStatusChange();
    } catch (e) { notify((e as Error).message, true); } finally { setBusy(false); }
  };

  const doImport = async () => {
    if (!impFile) return notify("Choose a weights file (.pt)", true);
    const fd = new FormData();
    fd.append("weights", impFile);
    fd.append("image_version", imp.image_version);
    fd.append("label_version", imp.label_version);
    fd.append("note", imp.note);
    if (imp.map50) fd.append("map50", imp.map50);
    try { const m = await api.importModel(fd); notify(`Imported as ${m.name}`); setImpFile(null); load(); } catch (e) { notify((e as Error).message, true); }
  };

  return (
    <div className="page">
      <div className="models">
        <section className="card">
          <h2>Train a model</h2>
          <div className="field"><label>Image version</label>
            <select className="input" value={form.image_version} onChange={set("image_version")}>
              {iv.length === 0 && <option value="">none yet — create one in Versions</option>}
              {[...iv].reverse().map((v) => <option key={v.name} value={v.name}>{v.name} · {v.images} images</option>)}
            </select>
          </div>
          <div className="field" style={{ marginTop: 8 }}><label>Label version</label>
            <select className="input" value={form.label_version} onChange={set("label_version")}>
              <option value="">Snapshot current labels</option>
              {[...lv].reverse().map((v) => <option key={v.name} value={v.name}>{v.name} · {v.boxes} boxes</option>)}
            </select>
          </div>
          {!form.label_version && (
            <label className="row muted" style={{ marginTop: 6 }}>
              <input id="train-unopened" type="checkbox" checked={unopened} onChange={(e) => setUnopened(e.target.checked)} /> include images SAM3 labelled that nobody opened
            </label>
          )}

          {preview && (
            <div style={{ marginTop: 10 }}>
              <div className="row wrap">
                <span className="chip">{preview.split.train} train</span>
                <span className="chip">{preview.split.val} val</span>
                <span className="chip">{preview.split.test} test</span>
                {preview.split.gold > 0 && <span className="chip warn">★ {preview.split.gold} gold</span>}
                <span className="chip">{preview.boxes} boxes</span>
              </div>
              <div className="muted" style={{ marginTop: 6 }}>Labels: {sourcesText(preview.label_sources)}</div>
              {share > 0.5 && <div className="notice">Most labels come from SAM3, so mAP measures how well the model matches SAM3, not the real objects.</div>}
              {preview.split.train > 0 && (preview.split.val === 0 || preview.split.test === 0) && (
                <div className="notice">The {preview.split.val === 0 ? "val" : "test"} split is empty, so scores will be measured on training images and look better than they are.</div>
              )}
              {preview.split.train === 0 && <div className="notice">No training images: the two versions share no images in the train split.</div>}
            </div>
          )}
          {previewErr && <div className="notice">{previewErr}</div>}

          <div className="formgrid">
            <div className="field"><label>Base model</label>
              <select className="input" value={form.base} onChange={set("base")} disabled={!!form.parent}>
                <option>yolo26n.pt</option><option>yolo26s.pt</option>
              </select>
            </div>
            <div className="field"><label>Fine-tune from</label>
              <select className="input" value={form.parent} onChange={set("parent")}>
                <option value="">pretrained weights</option>
                {doneModels.map((m) => <option key={m.name} value={m.name}>{m.name}</option>)}
              </select>
            </div>
            <div className="field"><label>Epochs</label><input className="input" type="number" min={1} max={500} value={form.epochs} onChange={set("epochs")} /></div>
            <div className="field"><label>Image size</label><input className="input" type="number" step={32} min={160} max={1280} value={form.imgsz} onChange={set("imgsz")} /></div>
            <div className="field"><label>Batch</label><input className="input" type="number" min={1} max={64} value={form.batch} onChange={set("batch")} /></div>
          </div>
          <div className="field"><label>Note</label><input className="input" value={form.note} onChange={set("note")} placeholder="e.g. labels ≥ 0.7 only" /></div>
          <div className="row" style={{ marginTop: 12 }}>
            <button className="btn primary" onClick={start} disabled={flags.demo || busy || !!gpuBusy || !form.image_version || !preview || preview.split.train === 0}>
              Start training
            </button>
            {flags.demo && <span className="muted">Needs a GPU, so it runs on your machine.</span>}
            {gpuBusy && <span className="muted">GPU busy: {gpuBusy === "training" ? `training ${status?.training}` : "auto-label running"}</span>}
          </div>
          <p className="muted" style={{ marginBottom: 0 }}>Batch 8 at 640px fits a 4 GB GPU for the n model. Auto-label and training never run at the same time.</p>

          {!flags.demo && <details>
            <summary>Train elsewhere (Colab / Kaggle) and import</summary>
            <p className="muted">Download the dataset bundle from Versions (includes <code>train.py</code>), train, then upload <code>best.pt</code> here with the versions you used.</p>
            <div className="formgrid">
              <div className="field"><label>Image version</label>
                <select className="input" value={imp.image_version} onChange={(e) => setImp({ ...imp, image_version: e.target.value })}>{iv.map((v) => <option key={v.name}>{v.name}</option>)}</select></div>
              <div className="field"><label>Label version</label>
                <select className="input" value={imp.label_version} onChange={(e) => setImp({ ...imp, label_version: e.target.value })}>{lv.map((v) => <option key={v.name}>{v.name}</option>)}</select></div>
              <div className="field"><label>mAP50 (optional)</label><input className="input" type="number" step={0.01} min={0} max={1} value={imp.map50} onChange={(e) => setImp({ ...imp, map50: e.target.value })} /></div>
            </div>
            <div className="field"><label>Note</label><input className="input" value={imp.note} onChange={(e) => setImp({ ...imp, note: e.target.value })} /></div>
            <div className="row" style={{ marginTop: 10 }}>
              <input type="file" accept=".pt" onChange={(e) => setImpFile(e.target.files?.[0] ?? null)} />
              <button className="btn" onClick={doImport} disabled={!impFile || !imp.image_version || !imp.label_version}>Import</button>
            </div>
          </details>}
        </section>

        <div className="mlist">
          {models.length === 0 && <div className="empty">No models yet. Train one on the left.</div>}
          {models.map((m) => (
            <ModelCard key={m.name} m={m} notify={notify} onChanged={() => { load(); onStatusChange(); }} onTrainFrom={(name) => { setForm({ ...form, parent: name }); window.scrollTo(0, 0); }} />
          ))}
        </div>
      </div>
    </div>
  );
}
