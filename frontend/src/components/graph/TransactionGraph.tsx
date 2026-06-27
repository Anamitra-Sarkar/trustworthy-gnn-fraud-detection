"use client";

import { useEffect, useRef, useCallback } from "react";
import type { Core, EventObjectNode, LayoutOptions } from "cytoscape";
import type { GraphNode, GraphEdge } from "@/types";

interface TransactionGraphProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  onNodeSelect?: (node: GraphNode) => void;
  layout?: "cose" | "circle" | "grid";
}

function getRiskColor(score: number): string {
  if (score >= 0.8) return "#ef4444";
  if (score >= 0.6) return "#f97316";
  if (score >= 0.4) return "#eab308";
  return "#22c55e";
}

export default function TransactionGraph({
  nodes,
  edges,
  onNodeSelect,
  layout = "cose",
}: TransactionGraphProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);

  const initGraph = useCallback(async () => {
    if (!containerRef.current || nodes.length === 0) return;

    const cytoscape = (await import("cytoscape")).default;

    if (cyRef.current) {
      cyRef.current.destroy();
    }

    const elements = [
      ...nodes.map((n) => ({
        data: {
          id: n.id,
          label: n.label || n.id,
          risk_score: n.risk_score,
          prediction: n.prediction,
          degree: n.degree || 1,
        },
      })),
      ...edges.map((e, i) => ({
        data: {
          id: `e${i}`,
          source: e.source,
          target: e.target,
          weight: e.weight || 1,
        },
      })),
    ];

    const layoutOptions: LayoutOptions = {
      name: layout,
      animate: false,
      randomize: layout === "cose",
      nodeRepulsion: () => 8000,
      idealEdgeLength: () => 80,
      gravity: 0.3,
    };

    const cy = cytoscape({
      container: containerRef.current,
      elements,
      style: [
        {
          selector: "node",
          style: {
            "background-color": (ele: { data: (key: string) => number }) =>
              getRiskColor(ele.data("risk_score")),
            label: "data(label)",
            color: "#e2e8f0",
            "font-size": "10px",
            "text-valign": "bottom",
            "text-margin-y": 6,
            width: (ele: { data: (key: string) => number }) =>
              Math.max(20, Math.min(50, ele.data("degree") * 5 + 15)),
            height: (ele: { data: (key: string) => number }) =>
              Math.max(20, Math.min(50, ele.data("degree") * 5 + 15)),
            "border-width": 2,
            "border-color": "#1e293b",
          } as Record<string, unknown>,
        },
        {
          selector: "edge",
          style: {
            width: 1,
            "line-color": "#334155",
            opacity: 0.4,
            "curve-style": "bezier",
          },
        },
        {
          selector: "node:selected",
          style: {
            "border-width": 3,
            "border-color": "#6366f1",
            "overlay-color": "#6366f1",
            "overlay-padding": 4,
            "overlay-opacity": 0.15,
          },
        },
        {
          selector: "node:active",
          style: {
            "overlay-opacity": 0.1,
          },
        },
      ],
      layout: layoutOptions,
      minZoom: 0.2,
      maxZoom: 5,
    });

    cy.on("tap", "node", (evt: EventObjectNode) => {
      const nodeData = evt.target.data();
      if (onNodeSelect) {
        onNodeSelect({
          id: nodeData.id as string,
          label: nodeData.label as string,
          risk_score: nodeData.risk_score as number,
          prediction: nodeData.prediction as number,
        });
      }
    });

    cyRef.current = cy;
  }, [nodes, edges, layout, onNodeSelect]);

  useEffect(() => {
    initGraph();
    return () => {
      if (cyRef.current) {
        cyRef.current.destroy();
        cyRef.current = null;
      }
    };
  }, [initGraph]);

  return (
    <div
      ref={containerRef}
      className="h-full w-full rounded-lg bg-background"
      style={{ minHeight: "500px" }}
    />
  );
}
