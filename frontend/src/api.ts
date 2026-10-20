import type { Annotation, AnnotationBody, Diff, ImageVersion, Job, LabelVersion, Model, AutoBody, AutoState, OverviewItem, PredictResult, ProjectSummary, ServerStatus, TrainBody, TrainPreview } from "./types";

const ROOT = "/api";
let BASE = ROOT;

/** Everything except the dataset list talks to one dataset; switch it here. */
export function setDataset(slug: string | null) {
  BASE = slug ? `${ROOT}/p/${slug}` : ROOT;
  flags.dataset = slug;
}

async function req<T>(path: string, init?: RequestInit, base = BASE): Promise<T> {
  const res = await fetch(base + path, init);
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") msg = body.detail;
    } catch {}
    throw new Error(msg);
  }
  const text = await res.text();
  return (text ? JSON.parse(text) : undefined) as T;
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  classes: () => req<string[]>("/classes"),
  putClasses: (names: string[]) => req<string[]>("/classes", json("PUT", names)),
  overview: () => req<OverviewItem[]>("/overview"),
  upload: (files: File[]) => {
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f));
    return req<unknown[]>("/images", { method: "POST", body: fd });
  },
  annotations: (hash: string) => req<Annotation[]>(`/images/${hash}/annotations`),
  addAnnotation: (hash: string, body: AnnotationBody) => req<Annotation>(`/images/${hash}/annotations`, json("POST", body)),
  updateAnnotation: (id: number, body: AnnotationBody) => req<Annotation>(`/annotations/${id}`, json("PUT", body)),
  deleteAnnotation: (id: number) => req<void>(`/annotations/${id}`, { method: "DELETE" }),
  setReviewed: (hash: string, reviewed: boolean) => req<unknown>(`/images/${hash}/reviewed?reviewed=${reviewed}`, { method: "PUT" }),
  startAutolabel: (hashes?: string[], minConfidence?: number) =>
    req<Job>("/jobs/autolabel", json("POST", { ...(hashes ? { hashes } : {}), ...(minConfidence != null ? { min_confidence: minConfidence } : {}) })),
  view: (hash: string, minConfidence: number) => req<unknown>(`/images/${hash}/view?min_confidence=${minConfidence}`, { method: "POST" }),
  job: (id: string) => req<Job>(`/jobs/${id}`),
  imageVersions: () => req<ImageVersion[]>("/image-versions"),
  createImageVersion: (b: { note: string; seed: number; ratios: [number, number, number] }) => req<ImageVersion>("/image-versions", json("POST", b)),
  imageDiff: (a: string, b: string) => req<Diff>(`/image-versions/${a}/diff/${b}`),
  labelVersions: () => req<LabelVersion[]>("/label-versions"),
  createLabelVersion: (note: string, includeUnopened = false) => req<LabelVersion>("/label-versions", json("POST", { note, include_unopened: includeUnopened })),
  setGold: (hash: string, gold: boolean) => req<unknown>(`/images/${hash}/gold?gold=${gold}`, { method: "PUT" }),
  predict: (name: string, src: { file?: File; hash?: string }, conf = 0.05, imgsz?: number) => {
    const fd = new FormData();
    if (src.file) fd.append("file", src.file);
    if (src.hash) fd.append("hash", src.hash);
    fd.append("conf", String(conf));
    if (imgsz) fd.append("imgsz", String(imgsz));
    return req<PredictResult>(`/models/${name}/predict`, { method: "POST", body: fd });
  },
  autoState: () => req<AutoState | null>("/auto"),
  autoRun: (b: AutoBody) => req<AutoState>("/auto/run", json("POST", b)),
  autoCancel: (id: string) => req<AutoState>(`/auto/${id}/cancel`, { method: "POST" }),
  labelDiff: (a: string, b: string) => req<Diff>(`/label-versions/${a}/diff/${b}`),
  status: () => req<ServerStatus>("/status"),
  models: () => req<Model[]>("/models"),
  trainPreview: (iv: string, lv?: string, includeUnopened = false) =>
    req<TrainPreview>(`/train-preview?image_version=${encodeURIComponent(iv)}` + (lv ? `&label_version=${encodeURIComponent(lv)}` : "") + (includeUnopened ? "&include_unopened=true" : "")),
  train: (b: TrainBody) => req<Model>("/models/train", json("POST", b)),
  cancelModel: (name: string) => req<Model>(`/models/${name}/cancel`, { method: "POST" }),
  modelLog: (name: string) => req<{ log: string }>(`/models/${name}/log?tail=60`),
  importModel: (fd: FormData) => req<Model>("/models/import", { method: "POST", body: fd }),
};

/** Overridden by the static demo; the real app always talks to the server. */
export const flags = { demo: false, dataset: null as string | null };
export const urls = {
  image: (hash: string) => `${BASE}/images/${hash}/file`,
  thumb: (hash: string, size = 256) => `${BASE}/images/${hash}/thumb?size=${size}`,
  weights: (name: string) => `${BASE}/models/${name}/weights`,
  cover: (slug: string, hash: string) => `${ROOT}/p/${slug}/images/${hash}/thumb?size=320`,
  export: (imageVersion: string, labelVersion?: string, bundle = false) =>
    `${BASE}/export?image_version=${encodeURIComponent(imageVersion)}` + (labelVersion ? `&label_version=${encodeURIComponent(labelVersion)}` : "") + (bundle ? "&bundle=true" : ""),
};
export const weightsUrl = (name: string) => urls.weights(name);
export const imageUrl = (hash: string) => urls.image(hash);
export const thumbUrl = (hash: string, size = 256) => urls.thumb(hash, size);
export const exportUrl = (imageVersion: string, labelVersion?: string, bundle = false) => urls.export(imageVersion, labelVersion, bundle);

export const projectsApi = {
  list: () => req<ProjectSummary[]>("/projects", undefined, ROOT),
  create: (name: string) => req<ProjectSummary>("/projects", json("POST", { name }), ROOT),
};
export const coverUrl = (slug: string, hash: string) => urls.cover(slug, hash);
export const predictUrl = (name: string) => `${window.location.origin}${BASE}/models/${name}/predict`;
