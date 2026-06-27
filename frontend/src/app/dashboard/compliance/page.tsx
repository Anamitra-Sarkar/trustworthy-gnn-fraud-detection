"use client";

import { useState, useEffect } from "react";
import { Shield, FileText, ChevronRight, Plus, X, Loader2 } from "lucide-react";
import Header from "@/components/layout/Header";
import ReportViewer from "@/components/compliance/ReportViewer";
import { cn } from "@/lib/utils";
import { api } from "@/lib/api";

const riskColors: Record<string, string> = {
  critical: "border-red-500/30 bg-red-500/5",
  high: "border-orange-500/30 bg-orange-500/5",
  medium: "border-amber-500/30 bg-amber-500/5",
  low: "border-emerald-500/30 bg-emerald-500/5",
};

const riskBadge: Record<string, string> = {
  critical: "text-red-400 bg-red-400/10",
  high: "text-orange-400 bg-orange-400/10",
  medium: "text-amber-400 bg-amber-400/10",
  low: "text-emerald-400 bg-emerald-400/10",
};

const statusBadge: Record<string, string> = {
  pending: "text-amber-400",
  reviewing: "text-blue-400",
  resolved: "text-emerald-400",
  dismissed: "text-muted-foreground",
};

export default function CompliancePage() {
  const [selectedReport, setSelectedReport] = useState<Record<string, unknown> | null>(null);
  const [reports, setReports] = useState<Record<string, unknown>[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const data = await api.getEscalations();
        if (!alive) return;
        const escalated = Array.isArray(data) ? data : [];
        const reportsWithMeta = escalated.map((e: Record<string, unknown>, i: number) => ({
          id: e.id ?? `RPT-${String(i + 1).padStart(3, "0")}`,
          node_id: e.node_id ?? "Unknown",
          risk_classification: (e.priority === "critical" || e.priority === "high")
            ? "high" : "medium",
          prediction: ((e.risk_score as number) ?? 0) > 0.5 ? 1 : 0,
          confidence: Math.min(1, ((e.risk_score as number) ?? 0) + 0.3),
          risk_score: e.risk_score ?? 0,
          escalation_status: e.status ?? "pending",
          generated_at: e.created_at ?? new Date().toISOString(),
          summary: (e.reason as string) ?? "No summary available",
          key_metrics: {
            risk_score: (e.risk_score as number) ?? 0,
          },
        }));
        setReports(reportsWithMeta);
      } catch (e) {
        if (!alive) return;
        setError(e instanceof Error ? e.message : "Failed to load compliance reports");
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => { alive = false; };
  }, []);

  return (
    <>
      <Header title="Compliance Reports" />
      <div className="space-y-6 p-6">
        {loading ? (
          <div className="flex flex-col items-center justify-center py-16">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
            <p className="mt-3 text-sm text-muted-foreground">Loading compliance reports...</p>
          </div>
        ) : error ? (
          <div className="flex flex-col items-center justify-center py-16">
            <p className="text-sm text-destructive">{error}</p>
          </div>
        ) : (<>
        {/* Actions */}
        <div className="flex items-center justify-between">
          <p className="text-sm text-muted-foreground">
            {reports.length} compliance reports generated
          </p>
          <button className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary/90">
            <Plus className="h-4 w-4" />
            Generate Report
          </button>
        </div>

        <div className="flex gap-6">
          {/* Report Cards */}
          <div className="flex-1 space-y-4">
            {reports.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-16 text-center">
                <FileText className="mb-3 h-10 w-10 text-muted-foreground/40" />
                <p className="text-sm text-muted-foreground">No compliance reports yet</p>
              </div>
            ) : reports.map((report: Record<string, unknown>) => {
              const r = report as Record<string, string | number>;
              return (
              <button
                key={r.id as string}
                onClick={() => setSelectedReport(report)}
                className={cn(
                  "w-full rounded-xl border p-5 text-left transition-all hover:border-primary/30",
                  riskColors[r.risk_classification as string] ?? "",
                  selectedReport?.id === r.id && "ring-1 ring-primary/50"
                )}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-card">
                      <FileText className="h-5 w-5 text-primary" />
                    </div>
                    <div>
                      <p className="font-medium text-foreground">
                        {r.node_id as string}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {r.id as string}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span
                      className={cn(
                        "rounded-full px-2.5 py-0.5 text-xs font-medium capitalize",
                        riskBadge[r.risk_classification as string] ?? ""
                      )}
                    >
                      {r.risk_classification as string}
                    </span>
                    <ChevronRight className="h-4 w-4 text-muted-foreground" />
                  </div>
                </div>
                <div className="mt-3 flex items-center gap-4 text-xs text-muted-foreground">
                  <span className="flex items-center gap-1">
                    <Shield className="h-3 w-3" />
                    {((r.confidence as number) * 100).toFixed(0)}% confidence
                  </span>
                  <span
                    className={cn(
                      "capitalize",
                      statusBadge[r.escalation_status as string] ?? ""
                    )}
                  >
                    {r.escalation_status as string}
                  </span>
                  <span>
                    {new Date(r.generated_at as string).toLocaleDateString()}
                  </span>
                </div>
              </button>
              );
            })}
          </div>

          {/* Report Detail */}
          {selectedReport && (
            <div className="glass w-[440px] shrink-0 rounded-xl p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-medium text-muted-foreground">Report Detail</h3>
                <button
                  onClick={() => setSelectedReport(null)}
                  className="rounded-lg p-1 text-muted-foreground hover:bg-secondary"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
              <ReportViewer report={selectedReport as never} />
            </div>
          )}
        </div>
        </>)}
      </div>
    </>
  );
}
