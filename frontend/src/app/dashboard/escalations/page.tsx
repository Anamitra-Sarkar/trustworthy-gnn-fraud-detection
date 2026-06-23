"use client";

import { useState } from "react";
import {
  AlertTriangle,
  Clock,
  CheckCircle2,
  XCircle,
  Filter,
} from "lucide-react";
import Header from "@/components/layout/Header";
import { cn } from "@/lib/utils";
import type { Escalation } from "@/types";

const mockEscalations: Escalation[] = [
  {
    id: "ESC-001",
    node_id: "TXN-2003",
    risk_score: 0.95,
    reason:
      "Critical fraud probability with multi-hop suspicious connections in transaction graph",
    status: "open",
    priority: "critical",
    created_at: new Date(Date.now() - 1800000).toISOString(),
    updated_at: new Date(Date.now() - 1800000).toISOString(),
    assigned_to: "Senior Analyst",
  },
  {
    id: "ESC-002",
    node_id: "TXN-2007",
    risk_score: 0.82,
    reason:
      "High risk transaction with links to previously flagged accounts in cluster analysis",
    status: "reviewing",
    priority: "high",
    created_at: new Date(Date.now() - 7200000).toISOString(),
    updated_at: new Date(Date.now() - 3600000).toISOString(),
    assigned_to: "Compliance Team",
  },
  {
    id: "ESC-003",
    node_id: "TXN-2009",
    risk_score: 0.67,
    reason:
      "Moderate risk with elevated uncertainty - model confidence below threshold for auto-clear",
    status: "open",
    priority: "medium",
    created_at: new Date(Date.now() - 14400000).toISOString(),
    updated_at: new Date(Date.now() - 14400000).toISOString(),
  },
  {
    id: "ESC-004",
    node_id: "TXN-2012",
    risk_score: 0.55,
    reason:
      "Borderline classification with conflicting evidential signals between graph and feature-based analysis",
    status: "reviewing",
    priority: "medium",
    created_at: new Date(Date.now() - 28800000).toISOString(),
    updated_at: new Date(Date.now() - 14400000).toISOString(),
    assigned_to: "ML Team",
  },
  {
    id: "ESC-005",
    node_id: "TXN-2001",
    risk_score: 0.91,
    reason:
      "Confirmed fraud pattern matching historical laundering scheme topology",
    status: "resolved",
    priority: "critical",
    created_at: new Date(Date.now() - 86400000).toISOString(),
    updated_at: new Date(Date.now() - 43200000).toISOString(),
    assigned_to: "Senior Analyst",
  },
  {
    id: "ESC-006",
    node_id: "TXN-2014",
    risk_score: 0.42,
    reason: "Low-confidence flagging, likely false positive based on review",
    status: "resolved",
    priority: "low",
    created_at: new Date(Date.now() - 172800000).toISOString(),
    updated_at: new Date(Date.now() - 86400000).toISOString(),
  },
];

const priorityConfig: Record<
  string,
  { color: string; bg: string; icon: typeof AlertTriangle }
> = {
  critical: { color: "text-red-400", bg: "bg-red-400/10", icon: AlertTriangle },
  high: { color: "text-orange-400", bg: "bg-orange-400/10", icon: AlertTriangle },
  medium: { color: "text-amber-400", bg: "bg-amber-400/10", icon: Clock },
  low: { color: "text-emerald-400", bg: "bg-emerald-400/10", icon: CheckCircle2 },
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

  const filtered =
    statusFilter === "all"
      ? mockEscalations
      : mockEscalations.filter((e) => e.status === statusFilter);

  const counts = {
    all: mockEscalations.length,
    open: mockEscalations.filter((e) => e.status === "open").length,
    reviewing: mockEscalations.filter((e) => e.status === "reviewing").length,
    resolved: mockEscalations.filter((e) => e.status === "resolved").length,
  };

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
          {filtered.map((escalation) => {
            const priority = priorityConfig[escalation.priority];
            const status = statusConfig[escalation.status];
            const PriorityIcon = priority.icon;

            return (
              <div
                key={escalation.id}
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
                      <PriorityIcon className={cn("h-5 w-5", priority.color)} />
                    </div>
                    <div>
                      <div className="flex items-center gap-3">
                        <span className="font-mono text-sm font-medium text-foreground">
                          {escalation.node_id}
                        </span>
                        <span
                          className={cn(
                            "rounded-full px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider",
                            priority.bg,
                            priority.color
                          )}
                        >
                          {escalation.priority}
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
                        {escalation.reason}
                      </p>
                      <div className="mt-2 flex items-center gap-4 text-xs text-muted-foreground">
                        <span>
                          Risk: {(escalation.risk_score * 100).toFixed(0)}%
                        </span>
                        <span>{escalation.id}</span>
                        {escalation.assigned_to && (
                          <span>Assigned: {escalation.assigned_to}</span>
                        )}
                        <span>
                          {new Date(escalation.created_at).toLocaleString()}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Status Update Buttons */}
                  {escalation.status !== "resolved" && (
                    <div className="flex gap-2">
                      {escalation.status === "open" && (
                        <button className="rounded-lg border border-blue-500/30 px-3 py-1.5 text-xs font-medium text-blue-400 transition-colors hover:bg-blue-400/10">
                          Start Review
                        </button>
                      )}
                      <button className="rounded-lg border border-emerald-500/30 px-3 py-1.5 text-xs font-medium text-emerald-400 transition-colors hover:bg-emerald-400/10">
                        <CheckCircle2 className="mr-1 inline h-3 w-3" />
                        Resolve
                      </button>
                      <button className="rounded-lg border border-border px-3 py-1.5 text-xs font-medium text-muted-foreground transition-colors hover:bg-secondary">
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
