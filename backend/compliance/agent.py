import os
import json
from groq import Groq
from pydantic import TypeAdapter
from config import GROQ_API_KEY, GROQ_MODEL
from .schemas import UnifiedAgenticReport


class ComplianceAgent:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = GROQ_MODEL
        self.schema_adapter = TypeAdapter(UnifiedAgenticReport)
        self.json_schema = self._build_strict_schema()

    def _build_strict_schema(self) -> dict:
        raw = self.schema_adapter.json_schema()
        raw["additionalProperties"] = False
        if "properties" in raw:
            for prop_val in raw["properties"].values():
                if isinstance(prop_val, dict) and "properties" in prop_val:
                    prop_val["additionalProperties"] = False
        if "$defs" in raw:
            for def_val in raw["$defs"].values():
                if isinstance(def_val, dict):
                    def_val["additionalProperties"] = False
        return raw

    def generate_report(self, node_payload: dict) -> UnifiedAgenticReport:
        prompt = f"""You are an automated compliance agent running in an AML pipeline.
Analyze the following graph node metrics:
{json.dumps(node_payload, indent=2)}

Compute the Subjective Logic uncertainty metrics. Validate if the structural context
justifies the conformal prediction set width. Generate a structured report
adhering strictly to the provided schema.

Key rules:
- If vacuity_uncertainty > 0.7, set compliance_escalation_required to true
- If conformal_prediction_set contains multiple labels, set compliance_escalation_required to true
- If both evidence values are low (<0.5), classify as "Unresolved"
- Provide detailed compliance_rationalization explaining the risk assessment
"""

        chat_completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional financial risk agent. Output reports strictly adhering to the provided JSON schema."
                },
                {"role": "user", "content": prompt}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "AML_Risk_Analysis_Report",
                    "strict": True,
                    "schema": self.json_schema
                }
            },
            temperature=0.1,
        )

        raw_json = chat_completion.choices[0].message.content
        report = self.schema_adapter.validate_json(raw_json)
        return report

    def batch_reports(self, payloads: list) -> list:
        return [self.generate_report(p) for p in payloads]
