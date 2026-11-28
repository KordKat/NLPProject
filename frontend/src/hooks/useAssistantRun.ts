import { useCallback, useRef, useState } from "react";
import { api, streamAssistant } from "../services/api";
import type { ActivityEvent, Artifact, PendingApproval, ResearchMode, SourceRef } from "../types";

export interface Draft {
  userText: string;
  text: string;
  status: string;
  outcome?: string;
  activity: ActivityEvent[];
  sources: SourceRef[];
  artifacts: Artifact[];
  plan: { task_type?: string; main_question?: string; subquestions?: string[] } | null;
  error: string;
  cancelled: boolean;
}

const emptyDraft = (userText: string): Draft => ({
  userText,
  text: "",
  status: "Starting",
  activity: [],
  sources: [],
  artifacts: [],
  plan: null,
  error: "",
  cancelled: false,
});

function mergeBy<T>(list: T[], items: T[], key: (t: T) => unknown): T[] {
  const out = [...list];
  for (const it of items) {
    const i = out.findIndex((x) => key(x) === key(it));
    if (i >= 0) out[i] = { ...out[i], ...it };
    else out.push(it);
  }
  return out;
}

interface Callbacks {
  onFinished: () => void;
  onArtifact?: (a: Artifact) => void;
  onGraphUpdated?: () => void;
  onSources?: (s: SourceRef[]) => void;
}

export function useAssistantRun(projectId: string | undefined, cb: Callbacks) {
  const [draft, setDraft] = useState<Draft | null>(null);
  const [running, setRunning] = useState(false);
  const [approvals, setApprovals] = useState<PendingApproval[]>([]);
  const runIdRef = useRef<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const cbRef = useRef(cb);
  cbRef.current = cb;

  const handle = useCallback((ev: any) => {
    switch (ev.type) {
      case "accepted":
      case "run_started":
        runIdRef.current = ev.run_id;
        break;
      case "status":
        setDraft((d) => (d ? { ...d, status: ev.text } : d));
        break;
      case "plan":
        setDraft((d) => (d ? { ...d, plan: ev } : d));
        break;
      case "research_step":
      case "graph_updated":
        setDraft((d) => (d ? { ...d, activity: [...d.activity, ev], status: ev.text || d.status } : d));
        if (ev.type === "graph_updated") cbRef.current.onGraphUpdated?.();
        break;
      case "tool_call":
        setDraft((d) => (d ? { ...d, activity: [...d.activity, ev], status: `Running ${ev.name}` } : d));
        break;
      case "tool_result":
        setDraft((d) =>
          d
            ? {
                ...d,
                activity: d.activity.map((a) => (a.type === "tool_call" && a.id === ev.id ? { ...a, preview: ev.preview, ok: ev.ok, error: ev.error } : a)),
              }
            : d,
        );
        break;
      case "retrieval_sources":
        setDraft((d) => (d ? { ...d, sources: mergeBy(d.sources, (ev.sources || []).filter((s: SourceRef) => s.ref != null), (s) => s.ref) } : d));
        cbRef.current.onSources?.(ev.sources || []);
        break;
      case "artifact":
      case "image":
        if (ev.artifact) {
          setDraft((d) => (d ? { ...d, artifacts: mergeBy(d.artifacts, [ev.artifact], (a) => a.id) } : d));
          cbRef.current.onArtifact?.(ev.artifact);
        }
        break;
      case "approval_required":
        setApprovals((l) => [...l, { approval_id: ev.approval_id, tool: ev.tool, arguments: ev.arguments, reason: ev.reason }]);
        setDraft((d) => (d ? { ...d, status: `Waiting for your approval (${ev.tool})` } : d));
        break;
      case "approval_resolved":
        setApprovals((l) => l.filter((a) => a.approval_id !== ev.approval_id));
        setDraft((d) => (d ? { ...d, activity: [...d.activity, { type: "approval_resolved", approved: ev.approved, text: ev.approved ? "Action approved" : "Action rejected" }] } : d));
        break;
      case "text_delta":
        setDraft((d) => (d ? { ...d, text: d.text + ev.text, status: "Writing answer" } : d));
        break;
      case "clear_answer":
        setDraft((d) => (d ? { ...d, text: "" } : d));
        break;
      case "replace_answer":
        setDraft((d) => (d ? { ...d, text: ev.text } : d));
        break;
      case "error":
        setDraft((d) => (d ? { ...d, error: ev.text } : d));
        break;
      case "cancelled":
        setDraft((d) => (d ? { ...d, cancelled: true, status: "Cancelled" } : d));
        break;
      case "done":
        setDraft((d) => d ? {
          ...d,
          text: ev.text ?? d.text,
          status: ev.status === "completed" ? "Done" : ev.status === "partial" ? "Partially completed" : ev.status === "cancelled" ? "Cancelled" : "Failed",
          outcome: ev.status,
        } : d);
        break;
    }
  }, []);

  const send = useCallback(
    async (message: string, mode: ResearchMode, attachments: string[]) => {
      if (!projectId || running) return;
      const ctrl = new AbortController();
      abortRef.current = ctrl;
      setRunning(true);
      setApprovals([]);
      setDraft(emptyDraft(message));
      try {
        await streamAssistant(projectId, { message, research_mode: mode, attachments, signal: ctrl.signal, onEvent: handle });
      } catch (err: any) {
        if (err?.name !== "AbortError") setDraft((d) => (d ? { ...d, error: err?.message || String(err) } : d));
      } finally {
        setRunning(false);
        runIdRef.current = null;
        abortRef.current = null;
        setApprovals([]);
        cbRef.current.onFinished();
      }
    },
    [projectId, running, handle],
  );

  const cancel = useCallback(async () => {
    if (projectId && runIdRef.current) {
      try {
        await api.cancelRun(projectId, runIdRef.current);
        return;
      } catch {
        /* fall back to abort */
      }
    }
    abortRef.current?.abort();
  }, [projectId]);

  const decide = useCallback(
    async (approvalId: string, approve: boolean) => {
      if (!projectId) return;
      await api.decideApproval(projectId, approvalId, approve);
      setApprovals((l) => l.filter((a) => a.approval_id !== approvalId));
    },
    [projectId],
  );

  const clearDraft = useCallback(() => setDraft(null), []);

  return { draft, running, approvals, send, cancel, decide, clearDraft };
}
