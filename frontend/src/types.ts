export type Status = "proposed" | "accepted" | "rejected";

export interface OverviewItem {
  hash: string;
  filename: string;
  width: number;
  height: number;
  reviewed: boolean;
  accepted: number;
  proposed: number;
  rejected: number;
  auto: number;
  auto_done: boolean;
  gold: boolean;
}

export interface Annotation {
  id: number;
  image_hash: string;
  class_name: string;
  cx: number;
  cy: number;
  w: number;
  h: number;
  source: string;
  status: Status;
  confidence: number | null;
}

export type AnnotationBody = Omit<Annotation, "id" | "image_hash">;

export interface ImageVersion {
  name: string;
  parent: string | null;
  created_at: string;
  note: string;
  images: number;
  split: { train: number; val: number; test: number; gold: number };
}

export interface LabelVersion {
  name: string;
  parent: string | null;
  created_at: string;
  note: string;
  classes: string[];
  images: number;
  boxes: number;
  sources: Record<string, number>;
}

export interface Job {
  id: string;
  status: "running" | "loading model" | "done" | "failed";
  done: number;
  total: number;
  boxes: number;
  error: string | null;
}

export interface Diff {
  images_added?: string[];
  images_removed?: string[];
  labels_changed?: string[];
  classes_added?: string[];
  classes_removed?: string[];
}

export interface ModelMetrics {
  map50?: number;
  map50_95?: number;
  precision?: number;
  recall?: number;
  per_class_map50?: Record<string, number>;
  gold?: { map50: number; map50_95: number; precision: number; recall: number; images: number };
  train_seconds?: number;
  live?: Record<string, number>;
  evaluated_on?: string;
}

export interface Model {
  name: string;
  parent: string | null;
  created_at: string;
  note: string;
  image_version: string;
  label_version: string;
  base: string;
  config: { base?: string; epochs?: number; imgsz?: number; batch?: number };
  status: "running" | "done" | "failed" | "cancelled";
  epoch: number;
  epochs: number;
  metrics: ModelMetrics;
  error: string | null;
  imported: boolean;
  has_weights: boolean;
  inference?: InferenceStats | null;
}

export interface TrainPreview {
  images: number;
  split: { train: number; val: number; test: number; gold: number };
  boxes: number;
  label_sources: Record<string, number>;
  classes: string[];
  snapshot: boolean;
}

export interface ServerStatus {
  gpu_busy: "training" | "autolabel" | "auto" | null;
  training: string | null;
  training_dataset?: string | null;
}

export interface ProjectSummary {
  slug: string;
  name: string;
  images: number;
  reviewed: number;
  boxes: number;
  models: number;
  versions: number;
  cover: string | null;
  created_at: string | null;
}

export interface TrainBody {
  image_version: string;
  label_version?: string | null;
  base: string;
  epochs: number;
  imgsz: number;
  batch: number;
  note: string;
  parent?: string | null;
  include_unopened?: boolean;
}

export interface AutoState {
  id: string;
  status: "running" | "done" | "failed" | "cancelled";
  step: "label" | "versions" | "train" | "evaluate" | "done";
  cfg: { base: string; epochs: number; imgsz: number; batch: number; relabel_all: boolean };
  error: string | null;
  model: string | null;
  label: { done: number; total: number };
  versions: { image: string; label: string } | null;
  epoch: number;
  epochs: number;
  detail: string;
  metrics: ModelMetrics | null;
}

export interface AutoBody {
  base: string;
  epochs: number;
  imgsz: number;
  batch: number;
  relabel_all: boolean;
}

export interface InferenceStats {
  runs: number;
  warm_runs: number;
  p50_ms: number | null;
  p95_ms: number | null;
  mean_ms: number | null;
  device: string;
}

export interface Detection {
  class_name: string;
  confidence: number;
  box: { x1: number; y1: number; x2: number; y2: number };
}

export interface PredictResult {
  model: string;
  width: number;
  height: number;
  device: string;
  imgsz: number;
  total_ms: number;
  inference_ms: number;
  cold: boolean;
  detections: Detection[];
}
