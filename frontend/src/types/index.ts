export interface MCDropoutResult {
  mean_prediction: number;
  std_deviation: number;
  predictions: number[];
  confidence_interval: [number, number];
}

export interface ConformalResult {
  prediction_set: number[];
  confidence_level: number;
  set_size: number;
  coverage: number;
}

export interface EvidentialResult {
  belief: number;
  disbelief: number;
  uncertainty: number;
  base_rate: number;
  alpha: number[];
  dirichlet_strength: number;
}

export interface UncertaintyData {
  method: string;
  confidence: number;
  mc_dropout?: MCDropoutResult;
  conformal?: ConformalResult;
  evidential?: EvidentialResult;
}

export interface InferenceResult {
  node_id: string;
  prediction: number;
  probability: number;
  risk_score: number;
  label: string;
  backbone: string;
  uncertainty: UncertaintyData;
  timestamp: string;
}

export interface ComplianceReport {
  id: string;
  node_id: string;
  risk_classification: "critical" | "high" | "medium" | "low";
  prediction: number;
  confidence: number;
  risk_score: number;
  escalation_status: "pending" | "reviewing" | "resolved" | "dismissed";
  generated_at: string;
  summary: string;
  key_metrics: Record<string, number>;
}

export interface GraphNode {
  id: string;
  label: string;
  risk_score: number;
  prediction: number;
  features?: number[];
  degree?: number;
}

export interface GraphEdge {
  source: string;
  target: string;
  weight?: number;
}

export interface DemoData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  results: InferenceResult[];
  metadata: {
    dataset: string;
    total_nodes: number;
    total_edges: number;
    flagged_count: number;
    backbone: string;
  };
}

export interface Escalation {
  id: string;
  node_id: string;
  risk_score: number;
  reason: string;
  status: "open" | "reviewing" | "resolved" | "dismissed";
  priority: "critical" | "high" | "medium" | "low";
  created_at: string;
  updated_at: string;
  assigned_to?: string;
}
