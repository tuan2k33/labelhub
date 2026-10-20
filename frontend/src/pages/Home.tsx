import { useCallback, useEffect, useState } from "react";
import { coverUrl, projectsApi } from "../api";
import type { ProjectSummary } from "../types";

interface Props {
  notify: (msg: string, error?: boolean) => void;
  onOpen: (slug: string) => void;
}

export default function Home({ notify, onOpen }: Props) {
  const [projects, setProjects] = useState<ProjectSummary[] | null>(null);
  const [name, setName] = useState("");
  const [creating, setCreating] = useState(false);

  const load = useCallback(async () => {
    try { setProjects(await projectsApi.list()); } catch (e) { notify((e as Error).message, true); setProjects([]); }
  }, [notify]);
  useEffect(() => { load(); }, [load]);

  const create = async () => {
    if (!name.trim()) return;
    setCreating(true);
    try {
      const p = await projectsApi.create(name);
      setName("");
      onOpen(p.slug);
    } catch (e) { notify((e as Error).message, true); } finally { setCreating(false); }
  };

  return (
    <div className="page">
      <div className="toolbar">
        <h2 style={{ margin: 0 }}>Datasets</h2>
        <span className="spacer" />
        <input
          id="new-dataset"
          className="input"
          style={{ width: 240 }}
          placeholder="New dataset name, e.g. Warehouse forklifts"
          value={name}
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && create()}
        />
        <button className="btn primary" onClick={create} disabled={creating || !name.trim()}>Create dataset</button>
      </div>
      <p className="muted" style={{ marginTop: 0 }}>
        Each dataset has its own images, classes, versions and models, so a dataset of forklifts never mixes with one of street signs.
      </p>

      {projects === null ? (
        <div className="muted">Loading…</div>
      ) : projects.length === 0 ? (
        <div className="empty">
          <h2>No datasets yet</h2>
          <p>Name your first one above, then add images and classes.</p>
        </div>
      ) : (
        <div className="grid dsgrid">
          {projects.map((p) => (
            <button key={p.slug} className="tile ds" onClick={() => onOpen(p.slug)}>
              {p.cover ? <img src={coverUrl(p.slug, p.cover)} alt="" loading="lazy" /> : <div className="nocover">no images yet</div>}
              <div className="meta" style={{ flexDirection: "column", alignItems: "flex-start", gap: 4 }}>
                <b>{p.name}</b>
                <span className="muted">{p.images} images · {p.reviewed} reviewed · {p.boxes} boxes</span>
                <span className="muted">{p.versions} image versions · {p.models} models</span>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
