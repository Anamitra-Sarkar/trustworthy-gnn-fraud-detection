from pydantic import BaseModel, Field
from typing import List, Literal, Optional


class NodeStructuralDynamics(BaseModel):
    node_id: int = Field(description="Target node index under evaluation.")
    homophily_index: float = Field(description="Fraction of adjacent neighbors sharing identical feature clusters.")
    node_degree_density: int = Field(description="Total number of structural edges connected to this node.")
    degree_power_law_adherence: float = Field(description="Local metric tracking degree conformity to power-law scale.")


class ConformalPredictiveBoundaries(BaseModel):
    conformal_prediction_set: List[int] = Field(description="Prediction set containing correct labels with target confidence.")
    prediction_set_cardinality: int = Field(description="Size of the prediction set.")
    coverage_guarantee_valid: bool = Field(description="Indicates if transductive permutation equivariance is preserved.")
    quantile_threshold: float = Field(description="Empirical non-conformity threshold used during evaluation.")


class EvidentialUncertaintyMetrics(BaseModel):
    licit_evidence: float = Field(description="Evidence parameter for the licit transaction class.")
    fraud_evidence: float = Field(description="Evidence parameter for the illicit fraud class.")
    vacuity_uncertainty: float = Field(description="Epistemic uncertainty from lack of local graph training data.")
    dissonance_uncertainty: float = Field(description="Conflict uncertainty from overlapping or contradictory evidence.")


class MCDropoutMetrics(BaseModel):
    epistemic_uncertainty: float = Field(description="Model ignorance uncertainty.")
    aleatoric_uncertainty: float = Field(description="Intrinsic data noise uncertainty.")
    total_uncertainty: float = Field(description="Total predictive entropy.")
    num_passes: int = Field(description="Number of stochastic forward passes used.")


class UnifiedAgenticReport(BaseModel):
    target_node_id: int = Field(description="Unique node index of evaluated transaction.")
    structural_context: NodeStructuralDynamics
    conformal_boundaries: ConformalPredictiveBoundaries
    evidential_uncertainty: EvidentialUncertaintyMetrics
    mc_dropout_metrics: Optional[MCDropoutMetrics] = None
    final_risk_classification: Literal["Licit", "Fraud", "Unresolved"] = Field(
        description="Assigned risk class. Unresolved indicates high uncertainty."
    )
    compliance_escalation_required: bool = Field(
        description="True if vacuity exceeds 0.7 or conformal set contains multiple labels."
    )
    compliance_rationalization: str = Field(
        description="Detailed justification aligning UQ metrics and structural context."
    )


class InferenceRequest(BaseModel):
    node_features: List[float] = Field(description="Node feature vector.")
    edge_list: Optional[List[List[int]]] = Field(default=None, description="Edge list [[src,dst],...].")
    model_name: str = Field(default="graphsage_original_elliptic", description="Model identifier.")
    uncertainty_method: Literal["conformal", "evidential", "mc_dropout", "all"] = Field(default="all")


class InferenceResponse(BaseModel):
    node_id: int
    prediction: str
    confidence: float
    risk_score: float
    uncertainty: dict
    model_name: str


class BatchInferenceRequest(BaseModel):
    nodes: Optional[List[dict]] = Field(default=None, description="List of {id, features} dicts. Alternative to feature_matrix.")
    edges: Optional[List[List[int]]] = Field(default=None, description="Edge list [[src,dst],...]. Empty if not provided.")
    feature_matrix: Optional[List[List[float]]] = Field(default=None, description="Flat feature matrix [[f1,f2,...],...]. Auto-converted to nodes if nodes not provided.")
    model_name: str = Field(default="graphsage_original_elliptic")
    uncertainty_method: Literal["conformal", "evidential", "mc_dropout", "all"] = Field(default="all")


class EscalationRequest(BaseModel):
    analysis_id: str
    node_ids: List[int]
    reason: str = Field(default="Automated uncertainty threshold exceeded")
