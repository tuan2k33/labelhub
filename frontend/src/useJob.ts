import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import type { Job } from "./types";

/** Starts the auto-label job and polls it until it finishes. */
export function useAutolabel(onFinished: () => void, onError: (msg: string) => void) {
  const [job, setJob] = useState<Job | null>(null);
  const timer = useRef<number | undefined>(undefined);
  const finished = useRef(onFinished);
  const failed = useRef(onError);
  finished.current = onFinished;
  failed.current = onError;

  const poll = useCallback((id: string) => {
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(async () => {
      try {
        const j = await api.job(id);
        setJob(j);
        if (j.status === "done") {
          finished.current();
          setJob(null);
        } else if (j.status === "failed") {
          failed.current(j.error ?? "auto-label failed");
          setJob(null);
        } else poll(id);
      } catch (e) {
        failed.current((e as Error).message);
        setJob(null);
      }
    }, 800);
  }, []);

  useEffect(() => () => window.clearTimeout(timer.current), []);

  const start = useCallback(
    async (hashes?: string[], minConfidence?: number) => {
      try {
        const j = await api.startAutolabel(hashes, minConfidence);
        setJob(j);
        poll(j.id);
      } catch (e) {
        failed.current((e as Error).message);
      }
    },
    [poll],
  );

  return { job, start, running: job !== null };
}
