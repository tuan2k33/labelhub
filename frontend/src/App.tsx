import { useCallback, useEffect, useState } from "react";
import { api, flags, projectsApi, setDataset } from "./api";
import ClassesDialog from "./components/ClassesDialog";
import Editor from "./pages/Editor";
import Home from "./pages/Home";
import Auto from "./pages/Auto";
import Inference from "./pages/Inference";
import Library from "./pages/Library";
import Models from "./pages/Models";
import Versions from "./pages/Versions";
import type { OverviewItem, ProjectSummary, ServerStatus } from "./types";

type Page = "images" | "versions" | "models" | "auto" | "inference";

function useSlug() {
  const [slug, setSlug] = useState(() => window.location.hash.slice(1) || null);
  useEffect(() => {
    const f = () => setSlug(window.location.hash.slice(1) || null);
    window.addEventListener("hashchange", f);
    return () => window.removeEventListener("hashchange", f);
  }, []);
  return slug;
}

const go = (slug: string | null) => { window.location.hash = slug ?? ""; };

function useToast() {
  const [toast, setToast] = useState<{ msg: string; error?: boolean } | null>(null);
  const notify = useCallback((msg: string, error = false) => {
    setToast({ msg, error });
    window.setTimeout(() => setToast((t) => (t?.msg === msg ? null : t)), error ? 6000 : 2500);
  }, []);
  return { toast, notify };
}

const Toast = ({ toast }: { toast: { msg: string; error?: boolean } | null }) =>
  toast ? <div className={`toast${toast.error ? " error" : ""}`} role="status">{toast.msg}</div> : null;

const DemoBanner = () =>
  flags.demo ? (
    <div className="demo-banner">
      <b>Interactive demo.</b> The first dataset has 14 COCO sample photos with real SAM3 boxes; four start unlabelled so you can run Auto-label. Create more datasets to see how they stay separate. Edits stay in this tab. Its versions and trained model come from a real 1,000-image run. Training, export and SAM3 itself run on your own machine.
    </div>
  ) : null;

export default function App() {
  const slug = useSlug();
  const { toast, notify } = useToast();
  if (slug) return <Workspace key={slug} slug={slug} notify={notify} toast={toast} />;
  setDataset(null);
  return (
    <div className="app">
      <header className="topbar">
        <span className="brand">LabelHub</span>
      </header>
      <DemoBanner />
      <main className="main"><Home notify={notify} onOpen={go} /></main>
      <Toast toast={toast} />
    </div>
  );
}

interface WsProps { slug: string; notify: (msg: string, error?: boolean) => void; toast: { msg: string; error?: boolean } | null }

function Workspace({ slug, notify, toast }: WsProps) {
  setDataset(slug);
  const [page, setPage] = useState<Page>("images");
  const [classes, setClasses] = useState<string[]>([]);
  const [items, setItems] = useState<OverviewItem[]>([]);
  const [datasets, setDatasets] = useState<ProjectSummary[]>([]);
  const [open, setOpen] = useState<{ hash: string; order: string[] } | null>(null);
  const [editingClasses, setEditingClasses] = useState(false);
  const [status, setStatus] = useState<ServerStatus | null>(null);

  const refresh = useCallback(async () => {
    try { setItems(await api.overview()); } catch (e) { notify((e as Error).message, true); }
  }, [notify]);
  const refreshStatus = useCallback(() => { api.status().then(setStatus).catch(() => {}); }, []);

  useEffect(() => {
    refreshStatus();
    const t = window.setInterval(refreshStatus, 4000);
    return () => window.clearInterval(t);
  }, [refreshStatus]);

  useEffect(() => {
    api.classes().then(setClasses).catch((e) => notify(e.message, true));
    projectsApi.list().then(setDatasets).catch(() => {});
    refresh();
  }, [refresh, notify]);

  const saveClasses = async (names: string[]) => {
    try {
      setClasses(await api.putClasses(names));
      setEditingClasses(false);
      notify("Classes saved");
    } catch (e) { notify((e as Error).message, true); }
  };

  const reviewed = items.filter((i) => i.reviewed).length;
  const tab = (p: Page, label: string) => (
    <button className={page === p ? "active" : ""} onClick={() => { setPage(p); setOpen(null); }}>{label}</button>
  );

  return (
    <div className="app">
      <header className="topbar">
        <a className="brand" href="#" onClick={(e) => { e.preventDefault(); go(null); }}>LabelHub</a>
        <span className="muted">/</span>
        <select
          id="dataset-switcher"
          className="input switcher"
          aria-label="Dataset"
          value={slug}
          onChange={(e) => (e.target.value === "__all" ? go(null) : go(e.target.value))}
        >
          {datasets.some((d) => d.slug === slug) ? null : <option value={slug}>{slug}</option>}
          {datasets.map((d) => <option key={d.slug} value={d.slug}>{d.name}</option>)}
          <option value="__all">All datasets…</option>
        </select>
        <nav className="nav">
          {tab("images", "Images")}
          {tab("versions", "Versions")}
          {tab("models", "Models")}
          {tab("auto", "Auto")}
          {tab("inference", "Inference")}
        </nav>
        <span className="spacer" />
        {status?.gpu_busy && (
          <span className="chip warn gpu">
            {status.gpu_busy === "training" ? `training ${status.training}${status.training_dataset && status.training_dataset !== slug ? ` (${status.training_dataset})` : ""}` : status.gpu_busy === "auto" ? "auto mode running" : "auto-labelling"}
          </span>
        )}
        <span className="muted" title="An image counts as reviewed once you open it">{reviewed}/{items.length} reviewed</span>
        <button className="btn" onClick={() => setEditingClasses(true)}>Classes ({classes.length})</button>
      </header>
      <DemoBanner />
      <main className="main" style={open ? { overflow: "hidden" } : undefined}>
        {open ? (
          <Editor
            key={open.hash}
            hash={open.hash}
            order={open.order}
            items={items}
            classes={classes}
            gpuBusy={status?.gpu_busy === "training"}
            notify={notify}
            onNav={(hash) => setOpen({ hash, order: open.order })}
            onBack={() => setOpen(null)}
            onChanged={refresh}
          />
        ) : page === "images" ? (
          <Library
            items={items}
            classes={classes}
            gpuBusy={status?.gpu_busy === "training"}
            notify={notify}
            onOpen={(hash, order) => setOpen({ hash, order })}
            onRefresh={refresh}
            onEditClasses={() => setEditingClasses(true)}
          />
        ) : page === "versions" ? (
          <Versions items={items} notify={notify} />
        ) : page === "models" ? (
          <Models status={status} notify={notify} onStatusChange={refreshStatus} />
        ) : page === "inference" ? (
          <Inference items={items} classes={classes} notify={notify} />
        ) : (
          <Auto items={items} classes={classes} status={status} notify={notify} onStatusChange={refreshStatus} onRefresh={refresh} onOpenModels={() => setPage("models")} />
        )}
      </main>
      {editingClasses && <ClassesDialog classes={classes} onSave={saveClasses} onClose={() => setEditingClasses(false)} />}
      <Toast toast={toast} />
    </div>
  );
}
