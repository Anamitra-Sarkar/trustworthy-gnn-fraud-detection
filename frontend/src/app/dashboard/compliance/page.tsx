"use client";

import { useState } from "react";
import { Shield, FileText, ChevronRight, Plus, X } from "lucide-react";
import Header from "@/components/layout/Header";
import ReportViewer from "@/components/compliance/ReportViewer";
import { cn } from "@/lib/utils";
import type { ComplianceReport } from "@/types";

const mockReports: ComplianceReport[] = [
  {
    id: "RPT-001",
    node_id: "TXN-2003",
    risk_classification: "critical",
    prediction: 1,
    confidence: 0.94,
    risk_score: 0.92,
    escalation_status: "pending",
    generated_at: new Date(Date.now() - 3600000).toISOString(),
    summary:
      "High-confidence fraud detection with strong evidential support. Transaction exhibits anomalous graph centrality patterns consistent with known money laundering schemes. Dirichlet strength indicates low model uncertainty.",
    key_metrics: {
      dirichlet_strength: 14.2,
      graph_centrality: 0.87,
      anomaly_score: 0.91,
    },
  },
  {
    id: "RPT-002",
    node_id: "TXN-2007",
    risk_classification: "high",
    prediction: 1,
    confidence: 0.82,
    risk_score: 0.78,
    escalation_status: "reviewing",
    generated_at: new Date(Date.now() - 7200000).toISOString(),
    summary:
      "Moderate-to-high fraud probability with elevated uncertainty. Graph neighborhood analysis reveals connections to previously flagged entities. Further review recommended.",
    key_metrics: {
      dirichlet_strength: 8.5,
      graph_centrality: 0.65,
      anomaly_score: 0.73,
    },
  },
  {
    id: "RPT-003",
    node_id: "TXN-2011",
    risk_classification: "medium",
    prediction: 0,
    confidence: 0.71,
    risk_score: 0.45,
    escalation_status: "resolved",
    generated_at: new Date(Date.now() - 86400000).toISOString(),
    summary:
      "Borderline classification with notable uncertainty. Node features within normal range but graph structure shows weak anomalous patterns. Cleared after manual review.",
    key_metrics: {
      dirichlet_strength: 5.1,
      graph_centrality: 0.42,
      anomaly_score: 0.38,
    },
  },
  {
    id: "RPT-004",
    node_id: "TXN-2015",
    risk_classification: "low",
    prediction: 0,
    confidence: 0.96,
    risk_score: 0.12,
    escalation_status: "dismissed",
    generated_at: new Date(Date.now() - 172800000).toISOString(),
    summary:
      "Low risk with high model confidence. Transaction patterns consistent with legitimate activity. No anomalous graph features detected.",
    key_metrics: {
      dirichlet_strength: 18.3,
      graph_centrality: 0.15,
      anomaly_score: 0.08,
    },
  },
];

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
  const [selectedReport, setSelectedReport] =
    useState<ComplianceReport | null>(null);

  return (
    <>
      <Header title="Compliance Reports" />
      <div className="space-y-6 p-6">
        {/* Actions */}
        <div className="flex items-center justify-between">
          <p className="text-sm text-muted-foreground">
            {mockReports.length} compliance reports generated
          </p>
          <button className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary/90">
            <Plus className="h-4 w-4" />
            Generate Report
          </button>
        </div>

        <div className="flex gap-6">
          {/* Report Cards */}
          <div className="flex-1 space-y-4">
            {mockReports.map((report) => (
              <button
                key={report.id}
                onClick={() => setSelectedReport(report)}
                className={cn(
                  "w-full rounded-xl border p-5 text-left transition-all hover:border-primary/30",
                  riskColors[report.risk_classification],
                  selectedReport?.id === report.id && "ring-1 ring-primary/50"
                )}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-card">
                      <FileText className="h-5 w-5 text-primary" />
                    </div>
                    <div>
                      <p className="font-medium text-foreground">
                        {report.node_id}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {report.id}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span
                      className={cn(
                        "rounded-full px-2.5 py-0.5 text-xs font-medium capitalize",
                        riskBadge[report.risk_classification]
                      )}
                    >
                      {report.risk_classification}
                    </span>
                    <ChevronRight className="h-4 w-4 text-muted-foreground" />
                  </div>
                </div>
                <div className="mt-3 flex items-center gap-4 text-xs text-muted-foreground">
                  <span className="flex items-center gap-1">
                    <Shield className="h-3 w-3" />
                    {(report.confidence * 100).toFixed(0)}% confidence
                  </span>
                  <span
                    className={cn(
                      "capitalize",
                      statusBadge[report.escalation_status]
                    )}
                  >
                    {report.escalation_status}
                  </span>
                  <span>
                    {new Date(report.generated_at).toLocaleDateString()}
                  </span>
                </div>
              </button>
            ))}
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
              <ReportViewer report={selectedReport} />
            </div>
          )}
        </div>
      </div>
    </>
  );
}
