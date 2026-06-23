"use client";

import { Shield, Clock, AlertTriangle } from "lucide-react";
import { cn } from "@/lib/utils";
import type { ComplianceReport } from "@/types";

interface ReportViewerProps {
  report: ComplianceReport;
}

const statusColors: Record<string, string> = {
  pending: "text-amber-400 bg-amber-400/10 border-amber-400/20",
  reviewing: "text-blue-400 bg-blue-400/10 border-blue-400/20",
  resolved: "text-emerald-400 bg-emerald-400/10 border-emerald-400/20",
  dismissed: "text-muted-foreground bg-muted/50 border-border",
};

const riskBadgeColors: Record<string, string> = {
  critical: "text-red-400 bg-red-400/10",
  high: "text-orange-400 bg-orange-400/10",
  medium: "text-amber-400 bg-amber-400/10",
  low: "text-emerald-400 bg-emerald-400/10",
};

export default function ReportViewer({ report }: ReportViewerProps) {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h3 className="text-lg font-semibold text-foreground">
            Report: {report.node_id}
          </h3>
          <div className="mt-1 flex items-center gap-2 text-xs text-muted-foreground">
            <Clock className="h-3 w-3" />
            {new Date(report.generated_at).toLocaleString()}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={cn(
              "rounded-full px-3 py-1 text-xs font-medium capitalize",
              riskBadgeColors[report.risk_classification]
            )}
          >
            {report.risk_classification}
          </span>
          <span
            className={cn(
              "rounded-full border px-3 py-1 text-xs font-medium capitalize",
              statusColors[report.escalation_status]
            )}
          >
            {report.escalation_status}
          </span>
        </div>
      </div>

      {/* Summary */}
      <div className="rounded-lg border border-border bg-secondary/30 p-4">
        <div className="mb-2 flex items-center gap-2 text-sm font-medium text-foreground">
          <Shield className="h-4 w-4 text-primary" />
          Summary
        </div>
        <p className="text-sm leading-relaxed text-muted-foreground">
          {report.summary}
        </p>
      </div>

      {/* Key Metrics */}
      <div>
        <h4 className="mb-3 flex items-center gap-2 text-sm font-medium text-foreground">
          <AlertTriangle className="h-4 w-4 text-primary" />
          Key Metrics
        </h4>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
          <div className="rounded-lg border border-border bg-secondary/20 p-3 text-center">
            <p className="text-lg font-bold text-foreground">
              {(report.confidence * 100).toFixed(1)}%
            </p>
            <p className="text-xs text-muted-foreground">Confidence</p>
          </div>
          <div className="rounded-lg border border-border bg-secondary/20 p-3 text-center">
            <p className="text-lg font-bold text-foreground">
              {(report.risk_score * 100).toFixed(1)}%
            </p>
            <p className="text-xs text-muted-foreground">Risk Score</p>
          </div>
          <div className="rounded-lg border border-border bg-secondary/20 p-3 text-center">
            <p className="text-lg font-bold text-foreground">
              {report.prediction === 1 ? "Fraud" : "Legit"}
            </p>
            <p className="text-xs text-muted-foreground">Prediction</p>
          </div>
          {Object.entries(report.key_metrics).map(([key, value]) => (
            <div
              key={key}
              className="rounded-lg border border-border bg-secondary/20 p-3 text-center"
            >
              <p className="text-lg font-bold text-foreground">
                {typeof value === "number" ? value.toFixed(3) : value}
              </p>
              <p className="truncate text-xs text-muted-foreground">{key}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
