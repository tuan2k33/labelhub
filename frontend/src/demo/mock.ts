import { api, flags, projectsApi, urls } from "../api";
import type { Annotation, AnnotationBody, Detection, Diff, ImageVersion, Job, LabelVersion, Model, OverviewItem, ProjectSummary, ServerStatus, TrainPreview } from "../types";
import raw from "./data.json";

interface Box { class_name: string; cx: number; cy: number; w: number; h: number; confidence: number }
interface Img { hash: string; filename: string; width: number; height: number; src: string; reviewed: boolean; autoDone: boolean; gold: boolean; pending: Box[] | null }
interface IvEntry { summary: ImageVersion; media?: string[]; assign?: Record<string, "train" | "val" | "test" | "gold">; digest?: string }
interface LvEntry { summary: LabelVersion; labels?: Record<string, number[][]>; digest?: string }
interface Dataset {
  slug: string; name: string; created_at: string; classes: string[]; images: Img[]; anns: Annotation[]; nextId: number;
  ivs: IvEntry[]; lvs: LvEntry[]; models: (Model & { log?: string })[]; uploads: number;
}

const DEMO_ONLY = "This demo has no GPU behind it. Training, export and SAM3 run on your machine; the models shown here come from a real run.";
const MIN_CONF = 0.5;
const data = raw as unknown as {
  classes: string[];
  images: (Omit<Img, "reviewed" | "pending"> & { boxes: Box[]; pending_autolabel: boolean })[];
  image_versions: ImageVersion[];
  label_versions: LabelVersion[];
  models: (Model & { log?: string })[];
  label_diffs: Record<string, Diff>;
  predictions: { model: string; imgsz: number; images: Record<string, { width: number; height: number; total_ms: number; inference_ms: number; detections: Detection[] }> };
};

const datasets = new Map<string, Dataset>();
const newDataset = (slug: string, name: string): Dataset => ({ slug, name, created_at: new Date().toISOString(), classes: [], images: [], anns: [], nextId: 1, ivs: [], lvs: [], models: [], uploads: 0 });
const cur = (): Dataset => {
  const d = datasets.get(flags.dataset ?? "");
  if (!d) throw new Error("dataset not found");
  return d;
};

// The sample dataset: 14 photos with real SAM3 boxes. Four start without boxes so Auto-label has work to do.
{
  const d = newDataset("coco-samples", "COCO street & home samples");
  d.classes = [...data.classes];
  d.images = data.images.map((i) => ({ hash: i.hash, filename: i.filename, width: i.width, height: i.height, src: i.src, reviewed: false, autoDone: !i.pending_autolabel, gold: false, pending: i.pending_autolabel ? i.boxes : null }));
  const add = (hash: string, b: Box) =>
    d.anns.push({ id: d.nextId++, image_hash: hash, class_name: b.class_name, cx: b.cx, cy: b.cy, w: b.w, h: b.h, source: "auto:sam3", status: b.confidence >= MIN_CONF ? "accepted" : "rejected", confidence: b.confidence });
  data.images.filter((i) => !i.pending_autolabel).forEach((i) => i.boxes.forEach((b) => add(i.hash, b)));
  d.ivs = data.image_versions.map((summary) => ({ summary }));
  d.lvs = data.label_versions.map((summary) => ({ summary }));
  d.models = data.models;
  datasets.set(d.slug, d);
}

let job: Job | null = null;
let jobTimer: number | undefined;

const wait = <T,>(v: T) => Promise.resolve(structuredClone(v));
const fail = (msg: string) => Promise.reject(new Error(msg));
const fnv = (s: string) => { let h = 2166136261; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return (h >>> 0) / 4294967296; };
const nextName = (names: string[]) => {
  const d = new Date();
  const prefix = `${d.getFullYear()}.${String(d.getMonth() + 1).padStart(2, "0")}.`;
  const used = names.filter((n) => n.startsWith(prefix)).map((n) => Number(n.slice(prefix.length)));
  return `${prefix}${Math.max(0, ...used) + 1}`;
};
const find = <T extends { summary: { name: string } }>(list: T[], name: string) => list.find((e) => e.summary.name === name);
const slugify = (n: string) => n.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 48);

function currentLabels(includeUnopened = false) {
  const { images, anns, classes } = cur();
  const index = new Map(classes.map((c, i) => [c, i]));
  const labels: Record<string, number[][]> = {};
  const sources: Record<string, number> = {};
  for (const im of images.filter((i) => i.reviewed || (includeUnopened && i.autoDone))) {
    const ok = anns.filter((a) => a.image_hash === im.hash && a.status === "accepted");
    const unknown = ok.filter((a) => !index.has(a.class_name)).map((a) => a.class_name);
    if (unknown.length) throw new Error(`annotations use unknown classes: ${[...new Set(unknown)].join(", ")}`);
    ok.forEach((a) => (sources[a.source] = (sources[a.source] ?? 0) + 1));
    labels[im.hash] = ok.map((a) => [index.get(a.class_name)!, +a.cx.toFixed(6), +a.cy.toFixed(6), +a.w.toFixed(6), +a.h.toFixed(6)]).sort((x, y) => JSON.stringify(x).localeCompare(JSON.stringify(y)));
  }
  return { labels, sources };
}

function overview(): OverviewItem[] {
  const { images, anns } = cur();
  return images.map((i) => {
    const mine = anns.filter((a) => a.image_hash === i.hash);
    const n = (s: string) => mine.filter((a) => a.status === s).length;
    return { hash: i.hash, filename: i.filename, width: i.width, height: i.height, reviewed: i.reviewed, accepted: n("accepted"), proposed: n("proposed"), rejected: n("rejected"), auto: mine.filter((a) => a.source.startsWith("auto:")).length, auto_done: i.autoDone, gold: i.gold };
  });
}

function readImage(file: File, n: number): Promise<Img> {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onerror = () => reject(new Error(`${file.name} could not be read`));
    fr.onload = () => {
      const im = new Image();
      im.onerror = () => reject(new Error(`${file.name} is not a valid image`));
      im.onload = () => resolve({ hash: `u${n}-${fnv(file.name + file.size).toString(16).slice(2, 8)}`, filename: file.name, width: im.naturalWidth, height: im.naturalHeight, src: String(fr.result), reviewed: false, autoDone: false, gold: false, pending: null });
      im.src = String(fr.result);
    };
    fr.readAsDataURL(file);
  });
}

const mock: typeof api = {
  classes: () => wait(cur().classes),
  putClasses: (names) => { cur().classes = [...names]; return wait(cur().classes); },
  overview: () => wait(overview()),
  upload: async (files) => {
    const d = cur();
    const added = await Promise.all(files.map((f) => readImage(f, ++d.uploads)));
    d.images.push(...added);
    return added.map((a) => ({ hash: a.hash }));
  },
  annotations: (hash) => wait(cur().anns.filter((a) => a.image_hash === hash)),
  addAnnotation: (hash, body: AnnotationBody) => { const d = cur(); const a = { ...body, id: d.nextId++, image_hash: hash } as Annotation; d.anns.push(a); return wait(a); },
  updateAnnotation: (id, body) => {
    const a = cur().anns.find((x) => x.id === id);
    if (!a) return fail("annotation not found");
    Object.assign(a, body);
    return wait(a);
  },
  deleteAnnotation: (id) => { const { anns } = cur(); const i = anns.findIndex((a) => a.id === id); if (i >= 0) anns.splice(i, 1); return Promise.resolve(); },
  setReviewed: (hash, reviewed) => { const im = cur().images.find((i) => i.hash === hash); if (im) im.reviewed = reviewed; return Promise.resolve({ hash, reviewed }); },
  setGold: (hash, gold) => {
    const im = cur().images.find((i) => i.hash === hash);
    if (gold && im && !im.reviewed) return fail("open and check the image before marking it gold: gold labels must be human-checked");
    if (im) im.gold = gold;
    return Promise.resolve({ hash, gold });
  },
  predict: (name, src, conf = 0.05, _imgsz) => {
    const base = data.predictions;
    const m = cur().models.find((x) => x.name === name);
    if (!m || m.status !== "done" || name !== base.model) return fail(`This demo can only run the real trained model ${base.model}.`);
    if (src.file) return fail("The demo has no GPU behind it, so it can only run the sample photos. Use “Pick from this dataset”.");
    const p = src.hash ? base.images[src.hash] : undefined;
    if (!p) return fail("Pick one of the 14 sample photos: these are the images the demo has real predictions for.");
    return wait({
      model: name, width: p.width, height: p.height, device: "0", imgsz: base.imgsz, total_ms: p.total_ms, inference_ms: p.inference_ms,
      cold: false, detections: p.detections.filter((d) => d.confidence >= conf),
    });
  },
  autoState: () => wait(null),
  autoRun: () => fail(DEMO_ONLY),
  autoCancel: () => fail(DEMO_ONLY),
  view: (hash, minConf) => {
    const d = cur();
    d.anns.filter((a) => a.image_hash === hash && a.status === "proposed").forEach((a) => { a.status = a.confidence == null || a.confidence >= minConf ? "accepted" : "rejected"; });
    const im = d.images.find((i) => i.hash === hash);
    if (im) im.reviewed = true;
    return Promise.resolve({ hash, reviewed: true });
  },

  startAutolabel: (hashes, minConf = MIN_CONF) => {
    if (job && job.status === "running") return fail("an auto-label job is already running");
    const d = cur();
    const todo = (hashes ?? d.images.filter((i) => !i.autoDone).map((i) => i.hash)).slice();
    job = { id: "demo", status: "running", done: 0, total: todo.length, boxes: 0, error: null };
    const j = job;
    window.clearInterval(jobTimer);
    const finish = () => { j.status = "done"; window.clearInterval(jobTimer); };
    if (!todo.length) { finish(); return wait(j); }
    jobTimer = window.setInterval(() => {
      const h = todo.shift();
      if (h === undefined) return finish();
      const im = d.images.find((i) => i.hash === h);
      const boxes = im?.pending ?? data.images.find((x) => x.hash === h)?.boxes ?? [];
      if (im) im.autoDone = true;
      if (im && boxes.length) {
        for (let k = d.anns.length - 1; k >= 0; k--) if (d.anns[k].image_hash === h && d.anns[k].source.startsWith("auto:")) d.anns.splice(k, 1);
        boxes.forEach((b) => d.anns.push({ id: d.nextId++, image_hash: h, class_name: b.class_name, cx: b.cx, cy: b.cy, w: b.w, h: b.h, source: "auto:sam3", status: b.confidence >= minConf ? "accepted" : "rejected", confidence: b.confidence }));
        j.boxes += boxes.length;
        im.pending = null;
      }
      j.done++;
      if (!todo.length) finish();
    }, 450);
    return wait(j);
  },
  job: () => wait(job ?? { id: "demo", status: "done", done: 0, total: 0, boxes: 0, error: null }),

  imageVersions: () => wait(cur().ivs.map((e) => e.summary)),
  createImageVersion: ({ note, seed, ratios }) => {
    const d = cur();
    const media = d.images.map((i) => i.hash).sort();
    if (!media.length) return fail("no images");
    const assign: Record<string, "train" | "val" | "test" | "gold"> = {};
    const golds = new Set(d.images.filter((i) => i.gold).map((i) => i.hash));
    media.forEach((h) => { const x = fnv(`${seed}:${h}`); assign[h] = golds.has(h) ? "gold" : x < ratios[0] ? "train" : x < ratios[0] + ratios[1] ? "val" : "test"; });
    const digest = JSON.stringify({ media, assign });
    const last = d.ivs[d.ivs.length - 1];
    if (last?.digest === digest) return fail(`no changes since ${last.summary.name}`);
    const split = { train: 0, val: 0, test: 0, gold: 0 };
    Object.values(assign).forEach((s) => split[s]++);
    const summary: ImageVersion = { name: nextName(d.ivs.map((e) => e.summary.name)), parent: last?.summary.name ?? null, created_at: new Date().toISOString(), note, images: media.length, split };
    d.ivs.push({ summary, media, assign, digest });
    return wait(summary);
  },
  imageDiff: (a, b) => {
    const x = find(cur().ivs, a), y = find(cur().ivs, b);
    if (!x?.media || !y?.media) return fail("These versions come from different datasets. Compare two versions you create in this demo.");
    const ma = new Set(x.media), mb = new Set(y.media);
    return wait({ images_added: y.media.filter((h) => !ma.has(h)), images_removed: x.media.filter((h) => !mb.has(h)) });
  },

  labelVersions: () => wait(cur().lvs.map((e) => e.summary)),
  createLabelVersion: (note, includeUnopened = false) => {
    const d = cur();
    if (!d.classes.length) return fail("no classes defined");
    let c;
    try { c = currentLabels(includeUnopened); } catch (e) { return fail((e as Error).message); }
    if (!Object.keys(c.labels).length) return fail(includeUnopened ? "no labelled images" : "no reviewed images");
    const digest = JSON.stringify({ classes: d.classes, labels: c.labels });
    const last = d.lvs[d.lvs.length - 1];
    if (last?.digest === digest) return fail(`no changes since ${last.summary.name}`);
    const summary: LabelVersion = {
      name: nextName(d.lvs.map((e) => e.summary.name)), parent: last?.summary.name ?? null, created_at: new Date().toISOString(), note, classes: [...d.classes],
      images: Object.keys(c.labels).length, boxes: Object.values(c.labels).reduce((n, b) => n + b.length, 0), sources: c.sources,
    };
    d.lvs.push({ summary, labels: c.labels, digest });
    return wait(summary);
  },
  labelDiff: (a, b) => {
    const x = find(cur().lvs, a), y = find(cur().lvs, b);
    if (x?.labels && y?.labels) {
      const same = (h: string) => JSON.stringify(x.labels![h]) === JSON.stringify(y.labels![h]);
      return wait<Diff>({
        images_added: Object.keys(y.labels).filter((h) => !(h in x.labels!)),
        images_removed: Object.keys(x.labels).filter((h) => !(h in y.labels!)),
        labels_changed: Object.keys(x.labels).filter((h) => h in y.labels! && !same(h)),
        classes_added: y.summary.classes.filter((c) => !x.summary.classes.includes(c)),
        classes_removed: x.summary.classes.filter((c) => !y.summary.classes.includes(c)),
      });
    }
    const counts = data.label_diffs[`${a}|${b}`] ?? data.label_diffs[`${b}|${a}`];
    if (!counts || !(x && !x.labels && y && !y.labels)) return fail("These versions come from different datasets. Compare two versions that are both from the real run, or both made in this demo.");
    const fill = (n: unknown) => Array.from({ length: Number(n) || 0 }, (_, i) => String(i));
    return wait<Diff>({ images_added: fill(counts.images_added), images_removed: fill(counts.images_removed), labels_changed: fill(counts.labels_changed), classes_added: [], classes_removed: [] });
  },

  status: () => wait<ServerStatus>({ gpu_busy: job?.status === "running" ? "autolabel" : null, training: null }),
  models: () => wait(cur().models.map(({ log: _log, ...m }) => m)),
  trainPreview: (iv, lv) => {
    const d = cur();
    const i = find(d.ivs, iv), l = lv ? find(d.lvs, lv) : d.lvs[d.lvs.length - 1];
    if (i && !i.media && l && !l.labels) {
      return wait<TrainPreview>({ images: i.summary.images, split: i.summary.split, boxes: l.summary.boxes, label_sources: l.summary.sources, classes: l.summary.classes, snapshot: !lv });
    }
    let labels = l?.labels, sources = l?.summary.sources ?? {}, cls = l?.summary.classes ?? d.classes;
    if (!lv) { try { const c = currentLabels(); labels = c.labels; sources = c.sources; cls = d.classes; } catch (e) { return fail((e as Error).message); } }
    const split = { train: 0, val: 0, test: 0, gold: 0 };
    let boxes = 0, n = 0;
    for (const [h, s] of Object.entries(i?.assign ?? {})) if (labels && h in labels) { split[s]++; boxes += labels[h].length; n++; }
    return wait<TrainPreview>({ images: n, split, boxes, label_sources: sources, classes: cls, snapshot: !lv });
  },
  train: () => fail(DEMO_ONLY),
  cancelModel: () => fail(DEMO_ONLY),
  modelLog: (name) => wait({ log: cur().models.find((m) => m.name === name)?.log ?? "" }),
  importModel: () => fail(DEMO_ONLY),
};

const summary = (d: Dataset): ProjectSummary => ({
  slug: d.slug, name: d.name, images: d.images.length, reviewed: d.images.filter((i) => i.reviewed).length,
  boxes: d.anns.filter((a) => a.status === "accepted").length, models: d.models.length, versions: d.ivs.length,
  cover: d.images[0]?.hash ?? null, created_at: d.created_at,
});

export function installDemo() {
  Object.assign(api, mock);
  Object.assign(projectsApi, {
    list: () => wait([...datasets.values()].map(summary)),
    create: (name: string) => {
      const base = slugify(name);
      if (!base) return fail("give the dataset a name (letters or digits)");
      let slug = base, n = 2;
      while (datasets.has(slug)) slug = `${base}-${n++}`;
      const d = newDataset(slug, name.trim());
      datasets.set(slug, d);
      return wait(summary(d));
    },
  });
  flags.demo = true;
  const src = (h: string) => cur().images.find((i) => i.hash === h)?.src ?? "";
  urls.image = src;
  urls.thumb = src;
  urls.cover = (slug: string, h: string) => datasets.get(slug)?.images.find((i) => i.hash === h)?.src ?? "";
  urls.weights = () => "#";
  urls.export = () => "#";
}
