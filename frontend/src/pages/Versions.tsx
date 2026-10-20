import { useCallback, useEffect, useState } from "react";
import { api, exportUrl, flags } from "../api";
import type { Diff, ImageVersion, LabelVersion, Model, OverviewItem } from "../types";

interface Props {
  items: OverviewItem[];
  notify: (msg: string, error?: boolean) => void;
}

const fmt = (iso: string) => new Date(iso).toLocaleString();

function UsedBy({ models }: { models: Model[] }) {
  if (!models.length) return null;
  return (
    <div className="row wrap" style={{ marginTop: 6 }}>
      <span className="muted">trained:</span>
      {models.map((m) => (
        <span key={m.name} className={`chip ${m.status === "done" ? "good" : m.status === "running" ? "warn" : m.status === "failed" ? "bad" : ""}`}>
          {m.name}{m.status !== "done" ? ` · ${m.status}` : m.metrics.map50 != null ? ` · mAP50 ${m.metrics.map50.toFixed(2)}` : ""}
        </span>
      ))}
    </div>
  );
}

function DiffView({ diff }: { diff: Diff }) {
  const parts: [string, number | undefined, string][] = [
    ["images added", diff.images_added?.length, "good"],
    ["images removed", diff.images_removed?.length, "bad"],
    ["labels changed", diff.labels_changed?.length, "warn"],
  ];
  const shown = parts.filter(([, n]) => n);
  return (
    <div className="row wrap" style={{ marginTop: 8 }}>
      {shown.length === 0 && !diff.classes_added?.length && !diff.classes_removed?.length && <span className="muted">No differences</span>}
      {shown.map(([label, n, tone]) => (
        <span key={label} className={`chip ${tone}`}>{n} {label}</span>
      ))}
      {diff.classes_added?.map((c) => <span key={"+" + c} className="chip good">+ class {c}</span>)}
      {diff.classes_removed?.map((c) => <span key={"-" + c} className="chip bad">− class {c}</span>)}
    </div>
  );
}

function Compare({ names, load }: { names: string[]; load: (a: string, b: string) => Promise<Diff> }) {
  const [a, setA] = useState("");
  const [b, setB] = useState("");
  const [diff, setDiff] = useState<Diff | null>(null);
  const [err, setErr] = useState("");
  useEffect(() => {
    setDiff(null);
    if (names.length >= 2) {
      setA(names[names.length - 2]);
      setB(names[names.length - 1]);
    }
  }, [names]);
  useEffect(() => {
    if (!a || !b || a === b) return setDiff(null);
    load(a, b).then((d) => { setDiff(d); setErr(""); }).catch((e) => setErr(e.message));
  }, [a, b, load]);
  if (names.length < 2) return null;
  return (
    <div style={{ marginTop: 12 }}>
      <h3>Compare</h3>
      <div className="row">
        <select className="input" value={a} onChange={(e) => setA(e.target.value)}>{names.map((n) => <option key={n}>{n}</option>)}</select>
        <span className="muted">→</span>
        <select className="input" value={b} onChange={(e) => setB(e.target.value)}>{names.map((n) => <option key={n}>{n}</option>)}</select>
      </div>
      {err ? <div className="chip bad" style={{ marginTop: 8 }}>{err}</div> : diff && <DiffView diff={diff} />}
    </div>
  );
}

export default function Versions({ items, notify }: Props) {
  const [iv, setIv] = useState<ImageVersion[]>([]);
  const [lv, setLv] = useState<LabelVersion[]>([]);
  const [ivNote, setIvNote] = useState("");
  const [lvNote, setLvNote] = useState("");
  const [unopened, setUnopened] = useState(false);
  const [seed, setSeed] = useState(0);
  const [split, setSplit] = useState({ train: 70, val: 10, test: 20 });
  const [exportIv, setExportIv] = useState("");
  const [exportLv, setExportLv] = useState("");
  const [models, setModels] = useState<Model[]>([]);

  const load = useCallback(async () => {
    try {
      const [a, b, c] = await Promise.all([api.imageVersions(), api.labelVersions(), api.models()]);
      setIv(a);
      setLv(b);
      setModels(c);
    } catch (e) {
      notify((e as Error).message, true);
    }
  }, [notify]);
  useEffect(() => { load(); }, [load]);
  useEffect(() => { if (!exportIv && iv.length) setExportIv(iv[iv.length - 1].name); }, [iv, exportIv]);

  const sum = split.train + split.val + split.test;
  const reviewed = items.filter((i) => i.reviewed).length;
  const labelled = items.filter((i) => i.reviewed || i.auto_done).length;

  const createIv = async () => {
    if (sum !== 100) return notify("Split must add up to 100%", true);
    try {
      const v = await api.createImageVersion({ note: ivNote, seed, ratios: [split.train / 100, split.val / 100, split.test / 100] });
      setIvNote("");
      notify(`Created image version ${v.name}`);
      load();
    } catch (e) { notify((e as Error).message, true); }
  };
  const createLv = async () => {
    try {
      const v = await api.createLabelVersion(lvNote, unopened);
      setLvNote("");
      notify(`Created label version ${v.name}`);
      load();
    } catch (e) { notify((e as Error).message, true); }
  };
  const num = (k: keyof typeof split) => (e: React.ChangeEvent<HTMLInputElement>) => setSplit({ ...split, [k]: Number(e.target.value) || 0 });

  return (
    <div className="page">
      <div className="cols">
        <section className="card">
          <h2>Image versions</h2>
          <p className="muted" style={{ marginTop: 0 }}>A frozen set of images with its train/val/test split. Editing labels never changes it.</p>
          <div className="field"><label>Note</label><input className="input" value={ivNote} onChange={(e) => setIvNote(e.target.value)} placeholder="e.g. first 1000 street photos" /></div>
          <div className="formgrid">
            <div className="field"><label>Train %</label><input className="input" type="number" min={0} max={100} value={split.train} onChange={num("train")} /></div>
            <div className="field"><label>Val %</label><input className="input" type="number" min={0} max={100} value={split.val} onChange={num("val")} /></div>
            <div className="field"><label>Test %</label><input className="input" type="number" min={0} max={100} value={split.test} onChange={num("test")} /></div>
            <div className="field"><label>Seed</label><input className="input" type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value) || 0)} /></div>
          </div>
          <div className="row">
            <button className="btn primary" onClick={createIv} disabled={items.length === 0}>Create from {items.length} images</button>
            {sum !== 100 && <span className="chip bad">split = {sum}%</span>}
          </div>
          <div className="vlist">
            {iv.length === 0 && <div className="muted">No image versions yet.</div>}
            {[...iv].reverse().map((v) => (
              <div className="vitem" key={v.name}>
                <div><b>{v.name}</b> {v.note && <span className="muted">· {v.note}</span>}<div className="muted">{fmt(v.created_at)}</div><UsedBy models={models.filter((m) => m.image_version === v.name)} /></div>
                <div style={{ textAlign: "right" }}><div>{v.images} images</div><div className="muted">{v.split.train}/{v.split.val}/{v.split.test}{v.split.gold ? ` + ${v.split.gold} gold` : ""}</div></div>
              </div>
            ))}
          </div>
          <Compare names={iv.map((v) => v.name)} load={api.imageDiff} />
        </section>

        <section className="card">
          <h2>Label versions</h2>
          <p className="muted" style={{ marginTop: 0 }}>A snapshot of accepted boxes on reviewed images, plus the class list. Models record which one they trained on.</p>
          <div className="field"><label>Note</label><input className="input" value={lvNote} onChange={(e) => setLvNote(e.target.value)} placeholder="e.g. after fixing bottles" /></div>
          <label className="row muted" style={{ marginTop: 10 }} title="Images SAM3 has labelled but nobody opened. This is what Auto mode trains on.">
            <input id="include-unopened" type="checkbox" checked={unopened} onChange={(e) => setUnopened(e.target.checked)} /> also images nobody opened ({labelled - reviewed})
          </label>
          <div className="row" style={{ marginTop: 10 }}>
            <button className="btn primary" onClick={createLv} disabled={(unopened ? labelled : reviewed) === 0}>Snapshot {unopened ? labelled : reviewed} images</button>
          </div>
          <div className="vlist">
            {lv.length === 0 && <div className="muted">No label versions yet. Review some images first.</div>}
            {[...lv].reverse().map((v) => (
              <div className="vitem" key={v.name}>
                <div><b>{v.name}</b> {v.note && <span className="muted">· {v.note}</span>}<div className="muted">{fmt(v.created_at)} · {v.classes.join(", ")}</div><UsedBy models={models.filter((m) => m.label_version === v.name)} /></div>
                <div style={{ textAlign: "right" }}><div>{v.images} images</div><div className="muted">{v.boxes} boxes</div></div>
              </div>
            ))}
          </div>
          <Compare names={lv.map((v) => v.name)} load={api.labelDiff} />
        </section>

        <section className="card">
          <h2>Export YOLO dataset</h2>
          <p className="muted" style={{ marginTop: 0 }}>Training data is the images in the image version that have labels in the label version.</p>
          <div className="field"><label>Image version</label>
            <select className="input" value={exportIv} onChange={(e) => setExportIv(e.target.value)}>{iv.map((v) => <option key={v.name}>{v.name}</option>)}</select>
          </div>
          <div className="field" style={{ marginTop: 8 }}><label>Label version</label>
            <select className="input" value={exportLv} onChange={(e) => setExportLv(e.target.value)}>
              <option value="">Snapshot current labels</option>
              {lv.map((v) => <option key={v.name}>{v.name}</option>)}
            </select>
          </div>
          <div style={{ marginTop: 12 }}>
            {exportIv && flags.demo ? <div className="row wrap"><button className="btn primary" disabled>Download zip</button><button className="btn" disabled>Bundle for Colab</button><span className="muted">Export runs on your machine.</span></div> : exportIv ? <div className="row wrap"><a className="btn primary" href={exportUrl(exportIv, exportLv || undefined)} download>Download zip</a><a className="btn" href={exportUrl(exportIv, exportLv || undefined, true)} download title="Adds train.py for Colab/Kaggle">Bundle for Colab</a></div> : <span className="muted">Create an image version first.</span>}
          </div>
        </section>
      </div>
    </div>
  );
}
