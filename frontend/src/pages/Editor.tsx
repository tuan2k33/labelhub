import type Konva from "konva";
import type { KonvaEventObject } from "konva/lib/Node";
import { useCallback, useEffect, useRef, useState } from "react";
import { Image as KImage, Label, Layer, Rect, Stage, Tag, Text, Transformer } from "react-konva";
import { api, imageUrl } from "../api";
import { useAutolabel } from "../useJob";
import type { Annotation, AnnotationBody, OverviewItem } from "../types";

const COLORS = ["#22c55e", "#f59e0b", "#3b82f6", "#ef4444", "#a855f7", "#14b8a6", "#ec4899", "#84cc16", "#f97316", "#06b6d4"];
const colorOf = (classes: string[], name: string) => {
  const i = classes.indexOf(name);
  return i < 0 ? "#9ca3af" : COLORS[i % COLORS.length];
};
const MIN_CONF = 0.5; // what auto-label accepts by default
const FLOOR = 0.3; // SAM3 returns nothing below this, so the slider starts here
const clamp = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), hi);
const toBody = (a: Annotation): AnnotationBody => ({
  class_name: a.class_name, cx: a.cx, cy: a.cy, w: a.w, h: a.h, source: a.source, status: a.status, confidence: a.confidence,
});
const sourceLabel = (a: Annotation) =>
  a.source === "human" ? "you" : `${a.source.replace("auto:", "")}${a.confidence != null ? ` ${a.confidence.toFixed(2)}` : ""}`;

const Icon = ({ d, children }: { d?: string; children?: React.ReactNode }) => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    {d ? <path d={d} /> : children}
  </svg>
);
const TrashIcon = () => <Icon d="M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14" />;
const EyeIcon = ({ off }: { off: boolean }) => (
  <Icon>
    <path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z" />
    <circle cx="12" cy="12" r="3" />
    {off && <path d="M3 3l18 18" />}
  </Icon>
);

function useImage(url: string) {
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  useEffect(() => {
    setImg(null);
    const i = new window.Image();
    i.onload = () => setImg(i);
    i.src = url;
    return () => { i.onload = null; };
  }, [url]);
  return img;
}

interface Props {
  hash: string;
  order: string[];
  items: OverviewItem[];
  classes: string[];
  gpuBusy?: boolean;
  notify: (msg: string, error?: boolean) => void;
  onNav: (hash: string) => void;
  onBack: () => void;
  onChanged: () => void;
}

interface Draft { x: number; y: number; w: number; h: number }

export default function Editor({ hash, order, items, classes, gpuBusy, notify, onNav, onBack, onChanged }: Props) {
  const info = items.find((i) => i.hash === hash);
  const idx = order.indexOf(hash);
  const img = useImage(imageUrl(hash));
  const iw = img?.naturalWidth ?? info?.width ?? 1;
  const ih = img?.naturalHeight ?? info?.height ?? 1;

  const [anns, setAnns] = useState<Annotation[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [cls, setCls] = useState(0);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [view, setView] = useState({ scale: 1, x: 0, y: 0 });
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [space, setSpace] = useState(false);
  const [hiddenIds, setHiddenIds] = useState<Set<number>>(new Set());
  const [thr, setThr] = useState(MIN_CONF);
  const [showHidden, setShowHidden] = useState(false);

  const wrap = useRef<HTMLDivElement>(null);
  const stageRef = useRef<Konva.Stage>(null);
  const trRef = useRef<Konva.Transformer>(null);
  const rects = useRef(new Map<number, Konva.Rect>());
  const drawStart = useRef<{ x: number; y: number } | null>(null);
  const draftRef = useRef<Draft | null>(null);
  const fit = useRef(1);
  const annsRef = useRef(anns);
  const thrInit = useRef(false);
  const thrTimer = useRef<number | undefined>(undefined);
  annsRef.current = anns;
  draftRef.current = draft;

  const reload = useCallback(async () => {
    try {
      setAnns(await api.annotations(hash));
    } catch (e) {
      notify((e as Error).message, true);
    }
  }, [hash, notify]);

  const auto = useAutolabel(
    () => {
      reload();
      onChanged();
      notify("Boxes ready");
    },
    (m) => notify(m, true),
  );

  // Opening an image counts as reviewing it.
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        if (info && (!info.reviewed || info.proposed > 0)) {
          await api.view(hash, MIN_CONF);
          onChanged();
        }
      } catch (e) {
        notify((e as Error).message, true);
      }
      if (alive) reload();
    })();
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hash]);

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    setSize({ w: el.clientWidth, h: el.clientHeight });
    return () => ro.disconnect();
  }, []);

  const fitView = useCallback(() => {
    if (!size.w || !size.h) return;
    const s = Math.min(size.w / iw, size.h / ih) * 0.97;
    fit.current = s;
    setView({ scale: s, x: (size.w - iw * s) / 2, y: (size.h - ih * s) / 2 });
  }, [size.w, size.h, iw, ih]);
  useEffect(fitView, [fitView]);

  useEffect(() => {
    const tr = trRef.current;
    if (!tr) return;
    const node = selected != null ? rects.current.get(selected) : undefined;
    tr.nodes(node && !hiddenIds.has(selected!) ? [node] : []);
    tr.getLayer()?.batchDraw();
  }, [selected, anns, showHidden, hiddenIds]);

  const persist = useCallback(async (id: number, patch: Partial<AnnotationBody>) => {
    const cur = annsRef.current.find((a) => a.id === id);
    if (!cur) return;
    const next = { ...cur, ...patch };
    setAnns((prev) => prev.map((a) => (a.id === id ? next : a)));
    try {
      await api.updateAnnotation(id, toBody(next));
      onChanged();
    } catch (e) {
      notify((e as Error).message, true);
      reload();
    }
  }, [notify, onChanged, reload]);

  // The slider stays at 0.50 (the auto-label default) unless this image was already filtered differently.
  useEffect(() => {
    if (thrInit.current || !anns.length) return;
    thrInit.current = true;
    const auto = anns.filter((a) => a.source.startsWith("auto:") && a.confidence != null);
    if (auto.every((a) => ((a.confidence as number) >= MIN_CONF) === (a.status === "accepted"))) return;
    const kept = auto.filter((a) => a.status === "accepted").map((a) => a.confidence as number);
    const cut = auto.filter((a) => a.status === "rejected").map((a) => a.confidence as number);
    setThr(clamp(kept.length ? Math.floor(Math.min(...kept) * 100) / 100 : Math.ceil(Math.max(...cut) * 100) / 100, FLOOR, 1));
  }, [anns]);

  const applyThr = useCallback((value: number) => {
    annsRef.current
      .filter((a) => a.source.startsWith("auto:") && a.confidence != null)
      .forEach((a) => {
        const want = (a.confidence as number) >= value ? "accepted" : "rejected";
        if (a.status !== want) persist(a.id, { status: want });
      });
  }, [persist]);
  const onThr = (v: number) => {
    setThr(v);
    window.clearTimeout(thrTimer.current);
    thrTimer.current = window.setTimeout(() => applyThr(v), 200);
  };
  useEffect(() => () => window.clearTimeout(thrTimer.current), []);

  const toggleAll = useCallback(() => {
    const all = annsRef.current;
    const live = all.filter((a) => a.status !== "rejected");
    setHiddenIds((prev) => (live.length > 0 && live.every((a) => prev.has(a.id)) ? new Set() : new Set(all.map((a) => a.id))));
  }, []);
  const toggleOne = (id: number) => setHiddenIds((prev) => { const n = new Set(prev); if (n.has(id)) n.delete(id); else n.add(id); return n; });

  const reclass = useCallback((id: number, name: string) => persist(id, { class_name: name }), [persist]);

  const remove = useCallback(async (id: number) => {
    setAnns((prev) => prev.filter((a) => a.id !== id));
    setSelected((s) => (s === id ? null : s));
    try {
      await api.deleteAnnotation(id);
      onChanged();
    } catch (e) {
      notify((e as Error).message, true);
      reload();
    }
  }, [notify, onChanged, reload]);

  const geometry = (x: number, y: number, w: number, h: number) => ({
    cx: (x + w / 2) / iw, cy: (y + h / 2) / ih, w: w / iw, h: h / ih,
  });

  const finishDraw = useCallback(async () => {
    const d = draftRef.current;
    drawStart.current = null;
    setDraft(null);
    if (!d || d.w * view.scale < 6 || d.h * view.scale < 6) return;
    if (!classes.length) return notify("Add a class first (top right)", true);
    try {
      const a = await api.addAnnotation(hash, {
        class_name: classes[Math.min(cls, classes.length - 1)], ...geometry(d.x, d.y, d.w, d.h),
        source: "human", status: "accepted", confidence: null,
      });
      setAnns((prev) => [...prev, a]);
      setSelected(a.id);
      onChanged();
    } catch (e) {
      notify((e as Error).message, true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [classes, cls, hash, iw, ih, view.scale, notify, onChanged]);

  useEffect(() => {
    window.addEventListener("mouseup", finishDraw);
    return () => window.removeEventListener("mouseup", finishDraw);
  }, [finishDraw]);

  const go = useCallback((d: number) => {
    const next = order[idx + d];
    if (next) onNav(next);
  }, [idx, order, onNav]);

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName) || e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.key === " ") { e.preventDefault(); return setSpace(true); }
      if (/^[1-9]$/.test(e.key)) {
        const i = Number(e.key) - 1;
        if (i < classes.length) {
          setCls(i);
          if (selected != null) reclass(selected, classes[i]);
        }
      } else if (e.key === "Delete" || e.key === "Backspace") { if (selected != null) remove(selected); }
      else if (e.key === "h") toggleAll();
      else if (e.key === "ArrowRight" || e.key === "Enter") go(1);
      else if (e.key === "ArrowLeft") go(-1);
      else if (e.key === "Escape") setSelected(null);
      else if (e.key === "f") fitView();
    };
    const up = (e: KeyboardEvent) => e.key === " " && setSpace(false);
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => { window.removeEventListener("keydown", down); window.removeEventListener("keyup", up); };
  }, [classes, selected, reclass, remove, go, fitView, toggleAll]);

  const onMouseDown = (e: KonvaEventObject<MouseEvent>) => {
    const stage = stageRef.current;
    if (!stage || space || e.evt.button !== 0) return;
    if (e.target !== stage && e.target.name() !== "bg") return;
    const p = stage.getRelativePointerPosition();
    if (!p) return;
    const x = clamp(p.x, 0, iw), y = clamp(p.y, 0, ih);
    drawStart.current = { x, y };
    setDraft({ x, y, w: 0, h: 0 });
    setSelected(null);
  };
  const onMouseMove = () => {
    const s = drawStart.current;
    const p = stageRef.current?.getRelativePointerPosition();
    if (!s || !p) return;
    const x = clamp(p.x, 0, iw), y = clamp(p.y, 0, ih);
    setDraft({ x: Math.min(s.x, x), y: Math.min(s.y, y), w: Math.abs(x - s.x), h: Math.abs(y - s.y) });
  };
  const onWheel = (e: KonvaEventObject<WheelEvent>) => {
    e.evt.preventDefault();
    const stage = stageRef.current;
    const pointer = stage?.getPointerPosition();
    if (!stage || !pointer) return;
    const old = view.scale;
    const ns = clamp(old * (e.evt.deltaY > 0 ? 1 / 1.12 : 1.12), fit.current * 0.5, fit.current * 40);
    const at = { x: (pointer.x - view.x) / old, y: (pointer.y - view.y) / old };
    setView({ scale: ns, x: pointer.x - at.x * ns, y: pointer.y - at.y * ns });
  };

  const onDragEnd = (a: Annotation, e: KonvaEventObject<DragEvent>) => {
    const n = e.target;
    const w = a.w * iw, h = a.h * ih;
    const x = clamp(n.x(), 0, iw - w), y = clamp(n.y(), 0, ih - h);
    n.position({ x, y });
    persist(a.id, geometry(x, y, w, h));
  };
  const onTransformEnd = (a: Annotation, e: KonvaEventObject<Event>) => {
    const n = e.target as Konva.Rect;
    const x = clamp(n.x(), 0, iw - 4), y = clamp(n.y(), 0, ih - 4);
    const w = Math.min(Math.max(4, n.width() * n.scaleX()), iw - x);
    const h = Math.min(Math.max(4, n.height() * n.scaleY()), ih - y);
    n.scaleX(1); n.scaleY(1); n.position({ x, y }); n.size({ width: w, height: h });
    persist(a.id, geometry(x, y, w, h));
  };

  const live = anns.filter((a) => a.status !== "rejected");
  const hidden = anns.filter((a) => a.status === "rejected");
  const drawn = (showHidden ? anns : live).filter((a) => !hiddenIds.has(a.id));
  const allHidden = live.length > 0 && live.every((a) => hiddenIds.has(a.id));
  const listed = [...(showHidden ? anns : live)].sort(
    (a, b) => (a.status === "rejected" ? 1 : 0) - (b.status === "rejected" ? 1 : 0) || (b.confidence ?? 1) - (a.confidence ?? 1) || a.id - b.id,
  );
  const sel = anns.find((a) => a.id === selected);
  const s = view.scale;

  return (
    <div className="editor">
      <div className="canvas-wrap" ref={wrap}>
        {size.w > 0 && (
          <Stage
            ref={stageRef}
            width={size.w}
            height={size.h}
            scaleX={s}
            scaleY={s}
            x={view.x}
            y={view.y}
            draggable={space}
            onDragEnd={(e) => e.target === stageRef.current && setView((v) => ({ ...v, x: e.target.x(), y: e.target.y() }))}
            onMouseDown={onMouseDown}
            onMouseMove={onMouseMove}
            onWheel={onWheel}
            style={{ cursor: space ? "grab" : "crosshair" }}
          >
            <Layer>
              {img && <KImage name="bg" image={img} width={iw} height={ih} />}
              {drawn.map((a) => {
                const color = colorOf(classes, a.class_name);
                const x = (a.cx - a.w / 2) * iw, y = (a.cy - a.h / 2) * ih;
                const isSel = a.id === selected;
                return (
                  <Rect
                    key={a.id}
                    ref={(n) => { if (n) rects.current.set(a.id, n); else rects.current.delete(a.id); }}
                    x={x} y={y} width={a.w * iw} height={a.h * ih}
                    stroke={color}
                    strokeWidth={2}
                    strokeScaleEnabled={false}
                    dash={a.status === "rejected" ? [2, 4] : undefined}
                    opacity={a.status === "rejected" ? 0.5 : 1}
                    fill={color + (isSel ? "40" : "1a")}
                    draggable={!space}
                    onClick={() => setSelected(a.id)}
                    onTap={() => setSelected(a.id)}
                    onDragMove={(e) => {
                      const n = e.target;
                      setAnns((prev) => prev.map((p) => (p.id === a.id ? { ...p, cx: (n.x() + a.w * iw / 2) / iw, cy: (n.y() + a.h * ih / 2) / ih } : p)));
                    }}
                    onDragEnd={(e) => onDragEnd(a, e)}
                    onTransformEnd={(e) => onTransformEnd(a, e)}
                  />
                );
              })}
              {drawn.map((a) => {
                const color = colorOf(classes, a.class_name);
                const y = (a.cy - a.h / 2) * ih;
                const tagH = 16 / s;
                return (
                  <Label key={`l${a.id}`} x={(a.cx - a.w / 2) * iw} y={y < tagH ? y : y - tagH} listening={false}>
                    <Tag fill={color} />
                    <Text text={`${a.class_name}${a.source !== "human" && a.confidence != null ? " " + a.confidence.toFixed(2) : ""}`} fontSize={11 / s} padding={2 / s} fill="#0b0f19" />
                  </Label>
                );
              })}
              {draft && (
                <Rect x={draft.x} y={draft.y} width={draft.w} height={draft.h} stroke={colorOf(classes, classes[cls] ?? "")} strokeWidth={2} strokeScaleEnabled={false} dash={[6, 4]} listening={false} />
              )}
              <Transformer
                ref={trRef}
                rotateEnabled={false}
                keepRatio={false}
                flipEnabled={false}
                anchorSize={9}
                anchorStroke="#2563eb"
                anchorFill="#fff"
                borderStroke="#2563eb"
                boundBoxFunc={(o, n) => (n.width < 6 || n.height < 6 ? o : n)}
              />
            </Layer>
          </Stage>
        )}
        <div className="hint">Drag to draw · scroll to zoom · hold Space to pan · F fit · 1–9 class · Del remove · H show/hide all boxes · ←/→ or Enter next image</div>
      </div>

      <aside className="side">
        <section>
          <div className="row">
            <button className="btn small" onClick={onBack}>← Grid</button>
            <span className="spacer" />
            <button className="btn small" onClick={() => go(-1)} disabled={idx <= 0} aria-label="Previous image">←</button>
            <span className="muted">{idx + 1} / {order.length}</span>
            <button className="btn small" onClick={() => go(1)} disabled={idx < 0 || idx >= order.length - 1} aria-label="Next image">→</button>
          </div>
          <div style={{ marginTop: 8, wordBreak: "break-all" }}>{info?.filename}</div>
          <div className="muted">{iw}×{ih}</div>
          <label className="row" style={{ marginTop: 8 }} title="Gold images are checked by you and used only to score models, never to train them">
            <input id="gold-toggle" type="checkbox" checked={!!info?.gold} disabled={!info?.reviewed} onChange={(e) => api.setGold(hash, e.target.checked).then(onChanged).catch((er) => notify((er as Error).message, true))} />
            <span>Gold image <span className="muted">· scoring only</span></span>
          </label>
        </section>

        <section>
          <h3>Class {sel ? "of selected box" : "for new boxes"}</h3>
          <div className="row wrap">
            {classes.length === 0 && <span className="muted">No classes yet — add them top right.</span>}
            {classes.map((c, i) => {
              const active = sel ? sel.class_name === c : cls === i;
              return (
                <button key={c} className={`classchip${active ? " active" : ""}`} onClick={() => { setCls(i); if (sel) reclass(sel.id, c); }}>
                  <span className="dot" style={{ background: COLORS[i % COLORS.length] }} />
                  {c} {i < 9 && <kbd>{i + 1}</kbd>}
                </button>
              );
            })}
          </div>
        </section>

        <section>
          <div className="row">
            <h3 style={{ margin: 0 }}>Annotations · {live.length}</h3>
            <span className="spacer" />
            <button className={`iconbtn${allHidden ? " off" : ""}`} onClick={toggleAll} aria-pressed={allHidden} aria-label={allHidden ? "Show all boxes" : "Hide all boxes"} title="Show/hide all boxes (H)">
              <EyeIcon off={allHidden} />
            </button>
          </div>
          <div className="row wrap" style={{ marginTop: 8 }}>
            <button className="btn small" disabled={auto.running || !!gpuBusy || !classes.length} onClick={() => auto.start([hash], MIN_CONF)} title="Redraws the SAM3 boxes for this image; boxes you drew yourself stay">
              {auto.running ? (auto.job?.status === "loading model" ? "Loading SAM3…" : "Working…") : "Detect with SAM3"}
            </button>
          </div>
          <div className="slider" title="Keeps SAM3 boxes at or above this confidence">
            <label htmlFor="conf-slider">Confidence ≥ <b>{thr.toFixed(2)}</b></label>
            <input id="conf-slider" type="range" min={FLOOR} max={1} step={0.01} value={thr} onChange={(e) => onThr(Number(e.target.value))} />
          </div>
          {hidden.length > 0 && (
            <label className="muted row" style={{ marginTop: 6 }}>
              <input type="checkbox" checked={showHidden} onChange={(e) => setShowHidden(e.target.checked)} /> show {hidden.length} below the threshold
            </label>
          )}
        </section>

        <div className="scroll">
          {listed.length === 0 && <div className="muted" style={{ padding: "0 14px 12px" }}>No boxes. Drag on the image to draw one.</div>}
          {listed.map((a) => (
            <div key={a.id} className={`annrow${a.id === selected ? " sel" : ""}${a.status === "rejected" || hiddenIds.has(a.id) ? " rejected" : ""}`} onClick={() => setSelected(a.id)}>
              <span className="dot" style={{ background: colorOf(classes, a.class_name) }} />
              <span className="grow">{a.class_name} <span className="muted">· {sourceLabel(a)}</span></span>
              <button className={`iconbtn${hiddenIds.has(a.id) ? " off" : ""}`} aria-pressed={hiddenIds.has(a.id)} aria-label={hiddenIds.has(a.id) ? "Show box" : "Hide box"} title={hiddenIds.has(a.id) ? "Show on image" : "Hide on image"} onClick={(e) => { e.stopPropagation(); toggleOne(a.id); }}>
                <EyeIcon off={hiddenIds.has(a.id)} />
              </button>
              {a.status === "rejected" && <button className="btn small" title="Keep this box in the labels" onClick={(e) => { e.stopPropagation(); persist(a.id, { status: "accepted" }); }}>Keep</button>}
              <button className="btn small danger" title="Delete (Del)" aria-label="Delete box" onClick={(e) => { e.stopPropagation(); remove(a.id); }}><TrashIcon /></button>
            </div>
          ))}
        </div>

        <section style={{ borderBottom: 0, borderTop: "1px solid var(--border)" }}>
          <button className="btn primary" style={{ width: "100%", justifyContent: "center" }} onClick={() => go(1)} disabled={idx < 0 || idx >= order.length - 1}>
            Next image <kbd>Enter</kbd>
          </button>
        </section>
      </aside>
    </div>
  );
}
