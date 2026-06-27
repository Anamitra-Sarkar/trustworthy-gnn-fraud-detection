"use client";

import { useState, useMemo, useEffect, useRef, useCallback } from "react";
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
  Loader2,
  FileSpreadsheet,
  AlertCircle,
} from "lucide-react";
import Header from "@/components/layout/Header";
import UncertaintyGauge from "@/components/uncertainty/UncertaintyGauge";
import EvidentialTriangle from "@/components/uncertainty/EvidentialTriangle";
import { cn } from "@/lib/utils";
import { api } from "@/lib/api";
import type { InferenceResult } from "@/types";

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
  const [results, setResults] = useState<InferenceResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const data = await api.getDemoData();
        if (!alive) return;
        setResults(data.results ?? []);
      } catch (e) {
        if (!alive) return;
        setError(e instanceof Error ? e.message : "Failed to load analysis data");
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => { alive = false; };
  }, []);

  const parseCSV = useCallback((text: string): number[][] => {
    const lines = text.trim().split("\n");
    return lines.map((line) =>
      line.split(",").map((v) => {
        const n = parseFloat(v.trim());
        if (isNaN(n)) throw new Error(`Non-numeric value in CSV: ${v.trim()}`);
        return n;
      })
    );
  }, []);

  const parseJSON = useCallback((text: string): number[][] => {
    const parsed = JSON.parse(text);
    const arr = Array.isArray(parsed) ? parsed : (parsed as Record<string, unknown>).features ?? (parsed as Record<string, unknown>).feature_matrix ?? [];
    if (!Array.isArray(arr)) throw new Error("JSON must contain an array or { features: [...] }");
    return arr.map((row: unknown) => {
      if (Array.isArray(row)) return row.map((v: unknown) => { const n = Number(v); if (isNaN(n)) throw new Error(`Non-numeric: ${v}`); return n; });
      throw new Error("Each row must be an array of numbers");
    });
  }, []);

  const handleFile = useCallback(async (file: File) => {
    setUploadError(null);
    setFileName(file.name);
    setUploading(true);
    try {
      const text = await file.text();
      let matrix: number[][];
      if (file.name.endsWith(".csv")) {
        matrix = parseCSV(text);
      } else if (file.name.endsWith(".json")) {
        matrix = parseJSON(text);
      } else {
        throw new Error("Unsupported file type. Use .csv or .json");
      }
      if (matrix.length === 0) throw new Error("Empty file");
      if (matrix.length > 1000) throw new Error("Max 1000 rows supported. Got " + matrix.length);
      const data = await api.batchInfer({ feature_matrix: matrix });
      const inferred = ((data as Record<string, unknown>).results ?? (data as Record<string, unknown>[])) as InferenceResult[];
      if (inferred.length > 0) {
        setResults(inferred);
        setMode("demo");
      } else {
        throw new Error("No results returned from inference");
      }
    } catch (e) {
      setUploadError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }, [parseCSV, parseJSON, setResults, setMode]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => setDragOver(false), []);

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
  };

  const sorted = useMemo(() => {
    return [...results].sort((a, b) => {
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
  }, [sortKey, sortDir, results]);

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

        {loading ? (
          <div className="flex flex-col items-center justify-center py-16">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
            <p className="mt-3 text-sm text-muted-foreground">Loading analysis data...</p>
          </div>
        ) : error ? (
          <div className="flex flex-col items-center justify-center py-16">
            <p className="text-sm text-destructive">{error}</p>
          </div>
        ) : (<>
        {mode === "upload" && (
          <div
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            className={cn(
              "flex flex-col items-center justify-center rounded-xl border-2 border-dashed p-12 transition-colors",
              dragOver
                ? "border-primary bg-primary/5"
                : "border-border",
              uploading && "pointer-events-none opacity-60"
            )}
          >
            {uploading ? (
              <>
                <Loader2 className="mb-3 h-10 w-10 animate-spin text-primary" />
                <p className="text-sm font-medium text-foreground">
                  Processing {fileName}...
                </p>
                <p className="mt-1 text-xs text-muted-foreground">
                  Running batch inference
                </p>
              </>
            ) : (
              <>
                <Upload className="mb-3 h-10 w-10 text-muted-foreground" />
                <p className="text-sm font-medium text-foreground">
                  {dragOver ? "Drop it!" : "Drop your CSV or JSON file here"}
                </p>
                <p className="mt-1 text-xs text-muted-foreground">
                  Feature matrix for batch inference (max 1000 rows)
                </p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv,.json"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) handleFile(file);
                  }}
                />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="mt-4 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary/90"
                >
                  Browse Files
                </button>
              </>
            )}
            {uploadError && (
              <div className="mt-4 flex items-center gap-2 rounded-lg bg-destructive/10 px-4 py-2 text-sm text-destructive">
                <AlertCircle className="h-4 w-4 shrink-0" />
                {uploadError}
              </div>
            )}
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
        </>)}
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
