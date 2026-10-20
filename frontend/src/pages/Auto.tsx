import { useCallback, useEffect, useState } from "react";
import { api, flags } from "../api";
import type { AutoState, OverviewItem, ServerStatus } from "../types";

interface Props {
  items: OverviewItem[];
  classes: string[];
  status: ServerStatus | null;
  notify: (msg: string, error?: boolean) => void;
  onStatusChange: () => void;
  onRefresh: () => void;
  onOpenModels: () => void;
}

const STEPS: { key: AutoState["step"]; title: string }[] = [
  { key: "label", title: "Label new images with SAM3" },
  { key: "versions", title: "Freeze the images and labels as versions" },
  { key: "train", title: "Train YOLO" },
  { key: "evaluate", title: "Score the model" },
];
const f3 = (n?: number) => (n == null ? "—" : n.toFixed(3));

function stepState(run: AutoState, i: number): "done" | "active" | "pending" | "failed" {
  const cur = run.step === "done" ? STEPS.length : STEPS.findIndex((s) => s.key === run.step);
  if (run.status === "done") return "done";
  if (i < cur) return "done";
  if (i === cur) return run.status === "running" ? "active" : "failed";
  return "pending";
}

export default function Auto({ items, classes, status, notify, onStatusChange, onRefresh, onOpenModels }: Props) {
  const [run, setRun] = useState<AutoState | null>(null);
  const [form, setForm] = useState({ base: "yolo26n.pt", epochs: 50, imgsz: 640, batch: 8, relabel_all: false });
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    try { setRun(await api.autoState()); } catch (e) { notify((e as Error).message, true); }
  }, [notify]);
  useEffect(() => { load(); }, [load]);

  const running = run?.status === "running";
  useEffect(() => {
    if (!running) return;
    const t = window.setInterval(() => { load(); onStatusChange(); }, 2000);
    return () => window.clearInterval(t);
  }, [running, load, onStatusChange]);
  useEffect(() => { if (run && run.status !== "running") onRefresh(); }, [run?.status]); // eslint-disable-line react-hooks/exhaustive-deps

  const fresh = items.filter((i) => !i.auto_done).length;
  const gold = items.filter((i) => i.gold).length;
  const gpuOther = status?.gpu_busy && status.gpu_busy !== "auto" && !running;
  const set = (k: keyof typeof form) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) =>
    setForm({ ...form, [k]: k === "relabel_all" ? (e.target as HTMLInputElement).checked : k === "base" ? e.target.value : Number(e.target.value) || 0 });

  const start = async () => {
    setBusy(true);
    try { setRun(await api.autoRun(form)); onStatusChange(); } catch (e) { notify((e as Error).message, true); } finally { setBusy(false); }
  };
  const cancel = async () => {
    if (!run) return;
    try { setRun(await api.autoCancel(run.id)); notify("Cancelling…"); } catch (e) { notify((e as Error).message, true); }
  };

  const detail = (key: AutoState["step"]) => {
    if (!run) return "";
    if (key === "label") return run.detail || (run.label.total === 0 ? "nothing new to label" : `${run.label.done} / ${run.label.total} images`);
    if (key === "versions") return run.versions ? `images ${run.versions.image} · labels ${run.versions.label}` : "";
    if (key === "train") return run.model ? `${run.model} · epoch ${run.epoch} / ${run.epochs}` : "";
    return run.metrics ? `mAP50 ${f3(run.metrics.map50)} against the SAM3 labels${run.metrics.gold ? `, ${f3(run.metrics.gold.map50)} on ${run.metrics.gold.images} gold images` : ""}` : "";
  };
  const pct = (key: AutoState["step"]) => {
    if (!run) return 0;
    if (key === "label") return run.label.total ? (run.label.done / run.label.total) * 100 : 100;
    if (key === "train") return run.epochs ? Math.min(100, (run.epoch / run.epochs) * 100) : 0;
    return 0;
  };

  return (
    <div className="page">
      <div className="autogrid">
        <section className="card">
          <h2>Auto mode</h2>
          <p className="muted" style={{ marginTop: 0 }}>
            No review step. SAM3 labels the images it has not seen, then YOLO trains on those labels. Check the result afterwards in Images and fix what you find.
          </p>
          <div className="row wrap">
            <span className="chip">{items.length} images</span>
            <span className="chip warn">{fresh} new to label</span>
            <span className="chip">{gold > 0 ? `★ ${gold} gold` : "no gold images"}</span>
            <span className="chip">{classes.length} classes</span>
          </div>
          {gold === 0 && (
            <div className="notice">
              Without gold images the score only shows how well YOLO copies SAM3. Open a few images, fix their boxes, and tick “Gold image” to get a score against labels you checked.
            </div>
          )}
          <div className="formgrid">
            <div className="field"><label>Base model</label>
              <select className="input" value={form.base} onChange={set("base")}><option>yolo26n.pt</option><option>yolo26s.pt</option></select></div>
            <div className="field"><label>Epochs</label><input className="input" type="number" min={1} max={500} value={form.epochs} onChange={set("epochs")} /></div>
            <div className="field"><label>Image size</label><input className="input" type="number" step={32} min={160} max={1280} value={form.imgsz} onChange={set("imgsz")} /></div>
            <div className="field"><label>Batch</label><input className="input" type="number" min={1} max={64} value={form.batch} onChange={set("batch")} /></div>
          </div>
          <label className="row muted">
            <input id="relabel-all" type="checkbox" checked={form.relabel_all} onChange={set("relabel_all")} /> label every image again, not just new ones
          </label>
          <div className="row" style={{ marginTop: 14 }}>
            <button className="btn primary" onClick={start} disabled={flags.demo || busy || running || !!gpuOther || classes.length === 0 || items.length === 0}>Run Auto mode</button>
            {running && <button className="btn danger" onClick={cancel}>Cancel</button>}
          </div>
          <p className="muted" style={{ marginBottom: 0 }}>
            {flags.demo ? "Needs a GPU, so it runs on your machine."
              : classes.length === 0 ? "Add classes first (top right)."
              : items.length === 0 ? "Add images first."
              : gpuOther ? `GPU busy: ${status?.gpu_busy}.`
              : "SAM3 is unloaded before training so both fit on a 4 GB card."}
          </p>
        </section>

        <section className="card">
          <h2>{run ? (running ? "Running" : run.status === "done" ? "Finished" : run.status === "cancelled" ? "Cancelled" : "Stopped") : "Last run"}</h2>
          {!run && <div className="muted">Nothing has run yet. Press “Run Auto mode” and the four steps appear here.</div>}
          {run && (
            <>
              <div className="steps">
                {STEPS.map((st, i) => {
                  const state = stepState(run, i);
                  return (
                    <div key={st.key} className={`step ${state}`}>
                      <span className="badge">{state === "done" ? "✓" : state === "failed" ? "!" : i + 1}</span>
                      <div>
                        <b>{st.title}</b>
                        <span className="muted">{detail(st.key)}</span>
                        {state === "active" && (st.key === "label" || st.key === "train") && (
                          <div className="progress" style={{ marginTop: 6 }}><span className="bar"><div style={{ width: `${pct(st.key)}%` }} /></span></div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
              {run.error && <div className="notice">{run.error}</div>}
              {run.status === "done" && run.metrics && (
                <>
                  <div className="mgrid">
                    <div className="metric"><div className="v">{f3(run.metrics.map50)}</div><div className="l">mAP50 vs SAM3 labels</div></div>
                    <div className="metric"><div className="v">{f3(run.metrics.gold?.map50)}</div><div className="l">mAP50 vs gold</div></div>
                    <div className="metric"><div className="v">{f3(run.metrics.precision)}</div><div className="l">precision</div></div>
                    <div className="metric"><div className="v">{f3(run.metrics.recall)}</div><div className="l">recall</div></div>
                  </div>
                  {!run.metrics.gold && <div className="muted">No gold images, so there is no score against checked labels.</div>}
                  <div className="row" style={{ marginTop: 10 }}><button className="btn" onClick={onOpenModels}>Open in Models</button></div>
                </>
              )}
            </>
          )}
        </section>
      </div>
    </div>
  );
}
