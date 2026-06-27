"use client";

import { useState, useEffect, useCallback } from "react";
import {
  ZoomIn,
  ZoomOut,
  Maximize2,
  GitBranch,
  Circle,
  Loader2,
  X,
  AlertCircle,
} from "lucide-react";
import Header from "@/components/layout/Header";
import TransactionGraph from "@/components/graph/TransactionGraph";
import UncertaintyGauge from "@/components/uncertainty/UncertaintyGauge";
import { cn } from "@/lib/utils";
import { api } from "@/lib/api";
import type { GraphNode, GraphEdge } from "@/types";

export default function GraphExplorerPage() {
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [edges, setEdges] = useState<GraphEdge[]>([]);
  const [layout, setLayout] = useState<"cose" | "circle">("cose");
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [loading, setLoading] = useState(true);
  const [riskFilter, setRiskFilter] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const data = await api.getDemoData();
        if (!alive) return;
        setNodes(data.nodes ?? []);
        setEdges(data.edges ?? []);
      } catch (e) {
        if (!alive) return;
        setError(e instanceof Error ? e.message : "Failed to load graph data");
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => { alive = false; };
  }, []);

  const handleNodeSelect = useCallback((node: GraphNode) => {
    setSelectedNode(node);
  }, []);

  const filteredNodes = nodes.filter((n) => n.risk_score >= riskFilter);
  const filteredNodeIds = new Set(filteredNodes.map((n) => n.id));
  const filteredEdges = edges.filter(
    (e) => filteredNodeIds.has(e.source) && filteredNodeIds.has(e.target)
  );

  return (
    <>
      <Header title="Graph Explorer" />
      <div className="relative flex flex-1">
        {/* Toolbar */}
        <div className="absolute left-4 top-4 z-20 flex flex-col gap-2">
          <div className="glass flex flex-col gap-1 rounded-lg p-1.5">
            <button
              className="rounded-md p-2 text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
              title="Zoom In"
            >
              <ZoomIn className="h-4 w-4" />
            </button>
            <button
              className="rounded-md p-2 text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
              title="Zoom Out"
            >
              <ZoomOut className="h-4 w-4" />
            </button>
            <button
              className="rounded-md p-2 text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
              title="Fit View"
            >
              <Maximize2 className="h-4 w-4" />
            </button>
          </div>

          <div className="glass flex flex-col gap-1 rounded-lg p-1.5">
            <button
              onClick={() => setLayout("cose")}
              className={cn(
                "rounded-md p-2 transition-colors",
                layout === "cose"
                  ? "bg-primary/20 text-primary"
                  : "text-muted-foreground hover:bg-secondary hover:text-foreground"
              )}
              title="Force-Directed"
            >
              <GitBranch className="h-4 w-4" />
            </button>
            <button
              onClick={() => setLayout("circle")}
              className={cn(
                "rounded-md p-2 transition-colors",
                layout === "circle"
                  ? "bg-primary/20 text-primary"
                  : "text-muted-foreground hover:bg-secondary hover:text-foreground"
              )}
              title="Circle"
            >
              <Circle className="h-4 w-4" />
            </button>
          </div>

          {/* Risk Filter */}
          <div className="glass rounded-lg p-3">
            <p className="mb-2 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
              Risk Filter
            </p>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={riskFilter}
              onChange={(e) => setRiskFilter(parseFloat(e.target.value))}
              className="w-20 accent-primary"
            />
            <p className="mt-1 text-xs text-muted-foreground">
              {(riskFilter * 100).toFixed(0)}%+
            </p>
          </div>
        </div>

        {/* Graph Canvas */}
        {loading ? (
          <div className="flex flex-1 items-center justify-center">
            <div className="flex flex-col items-center gap-3">
              <Loader2 className="h-8 w-8 animate-spin text-primary" />
              <p className="text-sm text-muted-foreground">
                Loading graph data...
              </p>
            </div>
          </div>
        ) : error ? (
          <div className="flex flex-1 items-center justify-center">
            <div className="flex flex-col items-center gap-3">
              <AlertCircle className="h-8 w-8 text-destructive" />
              <p className="text-sm text-destructive">{error}</p>
            </div>
          </div>
        ) : nodes.length === 0 ? (
          <div className="flex flex-1 items-center justify-center">
            <p className="text-sm text-muted-foreground">No graph data available</p>
          </div>
        ) : (
          <div className="flex-1">
            <TransactionGraph
              nodes={filteredNodes}
              edges={filteredEdges}
              onNodeSelect={handleNodeSelect}
              layout={layout}
            />
          </div>
        )}

        {/* Node Detail Panel */}
        {selectedNode && (
          <div className="animate-slide-in-right absolute right-0 top-0 z-20 h-full w-80 border-l border-border bg-card p-6">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-semibold text-foreground">
                Node Details
              </h3>
              <button
                onClick={() => setSelectedNode(null)}
                className="rounded-lg p-1.5 text-muted-foreground hover:bg-secondary"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <div className="mt-6 space-y-5">
              <div>
                <p className="text-xs text-muted-foreground">Node ID</p>
                <p className="mt-1 font-mono text-sm font-medium text-foreground">
                  {selectedNode.id}
                </p>
              </div>

              <div>
                <p className="text-xs text-muted-foreground">Label</p>
                <p className="mt-1 text-sm font-medium text-foreground">
                  {selectedNode.label}
                </p>
              </div>

              <div>
                <p className="text-xs text-muted-foreground">Prediction</p>
                <p
                  className={cn(
                    "mt-1 text-sm font-semibold",
                    selectedNode.prediction === 1
                      ? "text-red-400"
                      : "text-emerald-400"
                  )}
                >
                  {selectedNode.prediction === 1
                    ? "Fraudulent"
                    : "Legitimate"}
                </p>
              </div>

              <div className="flex justify-center">
                <UncertaintyGauge
                  value={1 - selectedNode.risk_score}
                  label="Confidence"
                />
              </div>

              <div>
                <p className="text-xs text-muted-foreground">Risk Score</p>
                <div className="mt-2 h-2 overflow-hidden rounded-full bg-secondary">
                  <div
                    className={cn(
                      "h-full rounded-full transition-all",
                      selectedNode.risk_score >= 0.8
                        ? "bg-red-500"
                        : selectedNode.risk_score >= 0.6
                          ? "bg-orange-500"
                          : selectedNode.risk_score >= 0.4
                            ? "bg-yellow-500"
                            : "bg-emerald-500"
                    )}
                    style={{
                      width: `${selectedNode.risk_score * 100}%`,
                    }}
                  />
                </div>
                <p className="mt-1 text-right text-xs text-muted-foreground">
                  {(selectedNode.risk_score * 100).toFixed(1)}%
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}
