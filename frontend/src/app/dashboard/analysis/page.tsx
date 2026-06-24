"use client";

import { useState, useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import {
  ChevronDown,
  ChevronUp,
  Upload,
  Database,
  ArrowUpDown,
} from "lucide-react";
import Header from "@/components/layout/Header";
import UncertaintyGauge from "@/components/uncertainty/UncertaintyGauge";
import EvidentialTriangle from "@/components/uncertainty/EvidentialTriangle";
import { cn } from "@/lib/utils";
import type { InferenceResult } from "@/types";

// Mock results data
const mockResults: InferenceResult[] = Array.from({ length: 20 }, (_, i) => {
  const riskScore = Math.random();
  const confidence = 0.6 + Math.random() * 0.4;
  return {
    node_id: `TXN-${2000 + i}`,
    prediction: riskScore > 0.55 ? 1 : 0,
    probability: confidence,
    risk_score: riskScore,
    label: riskScore > 0.55 ? "Fraud" : "Legitimate",
    backbone: ["GCN", "GAT", "GraphSAGE", "GIN", "GAT-v2"][
      Math.floor(Math.random() * 5)
    ],
    uncertainty: {
      method: "evidential",
      confidence,
      evidential: {
        belief: riskScore * 0.8,
        disbelief: (1 - riskScore) * 0.7,
        uncertainty: 0.1 + Math.random() * 0.2,
        base_rate: 0.5,
        alpha: [2 + Math.random() * 5, 2 + Math.random() * 5],
        dirichlet_strength: 4 + Math.random() * 10,
      },
    },
    timestamp: new Date(
      Date.now() - Math.random() * 86400000 * 7
    ).toISOString(),
  };
});

type SortKey = "node_id" | "prediction" | "probability" | "risk_score";

function SortHeader({
  label,
  field,
  onSort,
}: {
  label: string;
  field: SortKey;
  onSort: (key: SortKey) => void;
}) {
  return (
    <button
      onClick={() => onSort(field)}
      className="inline-flex items-center gap-1 text-xs font-medium uppercase tracking-wider text-muted-foreground hover:text-foreground"
    >
      {label}
      <ArrowUpDown className="h-3 w-3" />
    </button>
  );
}

export default function AnalysisPage() {
  const [mode, setMode] = useState<"demo" | "upload">("demo");
  const [expandedRow, setExpandedRow] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>("risk_score");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
  };

  const sorted = useMemo(() => {
    return [...mockResults].sort((a, b) => {
      const aVal = a[sortKey];
      const bVal = b[sortKey];
      if (typeof aVal === "string" && typeof bVal === "string") {
        return sortDir === "asc"
          ? aVal.localeCompare(bVal)
          : bVal.localeCompare(aVal);
      }
      return sortDir === "asc"
        ? (aVal as number) - (bVal as number)
        : (bVal as number) - (aVal as number);
    });
  }, [sortKey, sortDir]);

  return (
    <>
      <Header title="Analysis" />
      <div className="space-y-6 p-6">
        {/* Mode Toggle */}
        <div className="flex items-center gap-3">
          <div className="flex gap-1 rounded-lg bg-secondary p-1">
            <button
              onClick={() => setMode("demo")}
              className={cn(
                "flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-colors",
                mode === "demo"
                  ? "bg-primary text-white"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              <Database className="h-4 w-4" />
              Demo Data
            </button>
            <button
              onClick={() => setMode("upload")}
              className={cn(
                "flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-colors",
                mode === "upload"
                  ? "bg-primary text-white"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              <Upload className="h-4 w-4" />
              Upload
            </button>
          </div>
        </div>

        {mode === "upload" && (
          <div className="glass flex flex-col items-center justify-center rounded-xl border-2 border-dashed border-border p-12">
            <Upload className="mb-3 h-10 w-10 text-muted-foreground" />
            <p className="text-sm font-medium text-foreground">
              Drop your CSV or JSON file here
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              Feature matrix for batch inference
            </p>
            <button className="mt-4 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary/90">
              Browse Files
            </button>
          </div>
        )}

        {/* Results Table */}
        <div className="glass overflow-hidden rounded-xl">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-border">
                  <th className="px-4 py-3 text-left">
                    <SortHeader label="Node ID" field="node_id" onSort={handleSort} />
                  </th>
                  <th className="px-4 py-3 text-left">
                    <SortHeader label="Prediction" field="prediction" onSort={handleSort} />
                  </th>
                  <th className="px-4 py-3 text-left">
                    <SortHeader label="Confidence" field="probability" onSort={handleSort} />
                  </th>
                  <th className="px-4 py-3 text-left">
                    <SortHeader label="Risk Score" field="risk_score" onSort={handleSort} />
                  </th>
                  <th className="px-4 py-3 text-left">
                    <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                      Backbone
                    </span>
                  </th>
                  <th className="w-10 px-4 py-3" />
                </tr>
              </thead>
              <tbody>
                {sorted.map((result) => (
                  <AnalysisRow
                    key={result.node_id}
                    result={result}
                    isExpanded={expandedRow === result.node_id}
                    onToggle={() =>
                      setExpandedRow(
                        expandedRow === result.node_id
                          ? null
                          : result.node_id
                      )
                    }
                  />
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </>
  );
}

function AnalysisRow({
  result,
  isExpanded,
  onToggle,
}: {
  result: InferenceResult;
  isExpanded: boolean;
  onToggle: () => void;
}) {
  const beliefData = result.uncertainty.evidential
    ? [
        {
          name: "Belief",
          value: result.uncertainty.evidential.belief,
          fill: "#22c55e",
        },
        {
          name: "Disbelief",
          value: result.uncertainty.evidential.disbelief,
          fill: "#ef4444",
        },
        {
          name: "Uncertainty",
          value: result.uncertainty.evidential.uncertainty,
          fill: "#eab308",
        },
      ]
    : [];

  return (
    <>
      <tr
        className="border-b border-border/50 transition-colors hover:bg-secondary/30 cursor-pointer"
        onClick={onToggle}
      >
        <td className="px-4 py-3">
          <span className="font-mono text-sm text-foreground">
            {result.node_id}
          </span>
        </td>
        <td className="px-4 py-3">
          <span
            className={cn(
              "rounded-full px-2.5 py-0.5 text-xs font-medium",
              result.prediction === 1
                ? "bg-red-400/10 text-red-400"
                : "bg-emerald-400/10 text-emerald-400"
            )}
          >
            {result.label}
          </span>
        </td>
        <td className="px-4 py-3">
          <div className="flex items-center gap-2">
            <div className="h-1.5 w-16 overflow-hidden rounded-full bg-secondary">
              <div
                className="h-full rounded-full bg-primary"
                style={{ width: `${result.probability * 100}%` }}
              />
            </div>
            <span className="text-xs text-muted-foreground">
              {(result.probability * 100).toFixed(1)}%
            </span>
          </div>
        </td>
        <td className="px-4 py-3">
          <span
            className={cn(
              "text-sm font-medium",
              result.risk_score >= 0.8
                ? "text-red-400"
                : result.risk_score >= 0.6
                  ? "text-orange-400"
                  : result.risk_score >= 0.4
                    ? "text-amber-400"
                    : "text-emerald-400"
            )}
          >
            {(result.risk_score * 100).toFixed(1)}%
          </span>
        </td>
        <td className="px-4 py-3">
          <span className="rounded-md bg-secondary px-2 py-1 text-xs text-muted-foreground">
            {result.backbone}
          </span>
        </td>
        <td className="px-4 py-3">
          {isExpanded ? (
            <ChevronUp className="h-4 w-4 text-muted-foreground" />
          ) : (
            <ChevronDown className="h-4 w-4 text-muted-foreground" />
          )}
        </td>
      </tr>

      {isExpanded && (
        <tr className="border-b border-border/50 bg-secondary/20">
          <td colSpan={6} className="px-6 py-6">
            <div className="grid gap-6 md:grid-cols-3">
              {/* Confidence Gauge */}
              <div className="flex flex-col items-center">
                <UncertaintyGauge
                  value={result.uncertainty.confidence}
                  label="Model Confidence"
                />
              </div>

              {/* Evidential Triangle */}
              {result.uncertainty.evidential && (
                <div>
                  <EvidentialTriangle
                    belief={result.uncertainty.evidential.belief}
                    disbelief={result.uncertainty.evidential.disbelief}
                    uncertainty={result.uncertainty.evidential.uncertainty}
                  />
                </div>
              )}

              {/* Belief Bar Chart */}
              {beliefData.length > 0 && (
                <div>
                  <h4 className="mb-2 text-sm font-medium text-foreground">
                    Belief Distribution
                  </h4>
                  <ResponsiveContainer width="100%" height={150}>
                    <BarChart data={beliefData}>
                      <XAxis
                        dataKey="name"
                        stroke="#64748b"
                        fontSize={10}
                        tickLine={false}
                      />
                      <YAxis
                        stroke="#64748b"
                        fontSize={10}
                        tickLine={false}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "#0f172a",
                          border: "1px solid #1e293b",
                          borderRadius: "8px",
                          color: "#e2e8f0",
                          fontSize: "11px",
                        }}
                      />
                      <Bar dataKey="value" radius={[3, 3, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          </td>
        </tr>
      )}
    </>
  );
}
