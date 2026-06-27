"use client";

import { useState, useEffect } from "react";
import {
  AlertTriangle,
  Clock,
  CheckCircle2,
  XCircle,
  Filter,
  Loader2,
} from "lucide-react";
import Header from "@/components/layout/Header";
import { cn } from "@/lib/utils";
import { api } from "@/lib/api";

const priorityConfig: Record<
  string,
  { color: string; bg: string }
> = {
  critical: { color: "text-red-400", bg: "bg-red-400/10" },
  high: { color: "text-orange-400", bg: "bg-orange-400/10" },
  medium: { color: "text-amber-400", bg: "bg-amber-400/10" },
  low: { color: "text-emerald-400", bg: "bg-emerald-400/10" },
};

const statusConfig: Record<string, { label: string; color: string }> = {
  open: { label: "Open", color: "text-red-400 bg-red-400/10 border-red-400/20" },
  reviewing: {
    label: "Reviewing",
    color: "text-blue-400 bg-blue-400/10 border-blue-400/20",
  },
  resolved: {
    label: "Resolved",
    color: "text-emerald-400 bg-emerald-400/10 border-emerald-400/20",
  },
};

type StatusFilter = "all" | "open" | "reviewing" | "resolved";

export default function EscalationsPage() {
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
  const [escalations, setEscalations] = useState<Record<string, unknown>[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const data = await api.getEscalations();
        if (!alive) return;
        setEscalations(Array.isArray(data) ? data : []);
      } catch (e) {
        if (!alive) return;
        setError(e instanceof Error ? e.message : "Failed to load escalations");
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => { alive = false; };
  }, []);

  const filtered =
    statusFilter === "all"
      ? escalations
      : escalations.filter((e) => (e as Record<string, string>).status === statusFilter);

  const counts = {
    all: escalations.length,
    open: escalations.filter((e) => (e as Record<string, string>).status === "open").length,
    reviewing: escalations.filter((e) => (e as Record<string, string>).status === "reviewing").length,
    resolved: escalations.filter((e) => (e as Record<string, string>).status === "resolved").length,
  };

  if (loading) {
    return (
      <>
        <Header title="Escalations" />
        <div className="flex flex-col items-center justify-center py-16">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
          <p className="mt-3 text-sm text-muted-foreground">Loading escalations...</p>
        </div>
      </>
    );
  }

  if (error) {
    return (
      <>
        <Header title="Escalations" />
        <div className="flex flex-col items-center justify-center py-16">
          <AlertTriangle className="h-8 w-8 text-destructive" />
          <p className="mt-3 text-sm text-destructive">{error}</p>
        </div>
      </>
    );
  }

  return (
    <>
      <Header title="Escalations" />
      <div className="space-y-6 p-6">
        {/* Status Filter */}
        <div className="flex items-center gap-2">
          <Filter className="h-4 w-4 text-muted-foreground" />
          {(["all", "open", "reviewing", "resolved"] as StatusFilter[]).map(
            (status) => (
              <button
                key={status}
                onClick={() => setStatusFilter(status)}
                className={cn(
                  "rounded-lg px-3 py-1.5 text-sm font-medium capitalize transition-colors",
                  statusFilter === status
                    ? "bg-primary text-white"
                    : "bg-secondary text-muted-foreground hover:text-foreground"
                )}
              >
                {status}{" "}
                <span className="ml-1 text-xs opacity-70">
                  ({counts[status]})
                </span>
              </button>
            )
          )}
        </div>

        {/* Escalation Cards */}
        <div className="space-y-3">
          {filtered.map((esc) => {
            const e = esc as Record<string, string | number>;
            const priority = priorityConfig[e.priority as string] ?? priorityConfig.medium;
            const status = statusConfig[e.status as string] ?? statusConfig.resolved;

            return (
              <div
                key={e.id as string}
                className="glass rounded-xl p-5 transition-all hover:border-primary/20"
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-4">
                    <div
                      className={cn(
                        "flex h-10 w-10 items-center justify-center rounded-lg",
                        priority.bg
                      )}
                    >
                      <AlertTriangle className={cn("h-5 w-5", priority.color)} />
                    </div>
                    <div>
                      <div className="flex items-center gap-3">
                        <span className="font-mono text-sm font-medium text-foreground">
                          {e.node_id as string}
                        </span>
                        <span
                          className={cn(
                            "rounded-full px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider",
                            priority.bg,
                            priority.color
                          )}
                        >
                          {e.priority as string}
                        </span>
                        <span
                          className={cn(
                            "rounded-full border px-2.5 py-0.5 text-[10px] font-medium",
                            status.color
                          )}
                        >
                          {status.label}
                        </span>
                      </div>
                      <p className="mt-1.5 max-w-xl text-sm text-muted-foreground">
                        {e.reason as string}
                      </p>
                      <div className="mt-2 flex items-center gap-4 text-xs text-muted-foreground">
                        <span>
                          Risk: {((e.risk_score as number) * 100).toFixed(0)}%
                        </span>
                        <span>{e.id as string}</span>
                        {e.assigned_to && (
                          <span>Assigned: {e.assigned_to as string}</span>
                        )}
                        <span>
                          {new Date(e.created_at as string).toLocaleString()}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Status Update Buttons */}
                  {e.status !== "resolved" && (
                    <div className="flex gap-2">
                      {e.status === "open" && (
                        <button
                          onClick={async () => {
                            try {
                              await api.updateEscalation(e.id as string, { status: "reviewing" });
                              const data = await api.getEscalations();
                              setEscalations(Array.isArray(data) ? data : []);
                            } catch {}
                          }}
                          className="rounded-lg border border-blue-500/30 px-3 py-1.5 text-xs font-medium text-blue-400 transition-colors hover:bg-blue-400/10"
                        >
                          Start Review
                        </button>
                      )}
                      <button
                        onClick={async () => {
                          try {
                            await api.updateEscalation(e.id as string, { status: "resolved" });
                            const data = await api.getEscalations();
                            setEscalations(Array.isArray(data) ? data : []);
                          } catch {}
                        }}
                        className="rounded-lg border border-emerald-500/30 px-3 py-1.5 text-xs font-medium text-emerald-400 transition-colors hover:bg-emerald-400/10"
                      >
                        <CheckCircle2 className="mr-1 inline h-3 w-3" />
                        Resolve
                      </button>
                      <button
                        onClick={async () => {
                          try {
                            await api.updateEscalation(e.id as string, { status: "dismissed" });
                            const data = await api.getEscalations();
                            setEscalations(Array.isArray(data) ? data : []);
                          } catch {}
                        }}
                        className="rounded-lg border border-border px-3 py-1.5 text-xs font-medium text-muted-foreground transition-colors hover:bg-secondary"
                      >
                        <XCircle className="mr-1 inline h-3 w-3" />
                        Dismiss
                      </button>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {filtered.length === 0 && (
          <div className="flex flex-col items-center justify-center py-16 text-center">
            <CheckCircle2 className="mb-3 h-10 w-10 text-muted-foreground" />
            <p className="text-sm font-medium text-foreground">
              No escalations found
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              All escalations with this status have been cleared
            </p>
          </div>
        )}
      </div>
    </>
  );
}
