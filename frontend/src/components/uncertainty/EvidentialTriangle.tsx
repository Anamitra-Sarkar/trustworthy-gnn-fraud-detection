"use client";

interface EvidentialTriangleProps {
  belief: number;
  disbelief: number;
  uncertainty: number;
}

export default function EvidentialTriangle({
  belief,
  disbelief,
  uncertainty,
}: EvidentialTriangleProps) {
  const total = belief + disbelief + uncertainty || 1;
  const bPct = Math.round((belief / total) * 100);
  const dPct = Math.round((disbelief / total) * 100);
  const uPct = Math.round((uncertainty / total) * 100);

  return (
    <div className="flex flex-col gap-4">
      <h4 className="text-sm font-medium text-foreground">
        Evidential Decomposition
      </h4>

      {/* Bars */}
      <div className="space-y-3">
        <div>
          <div className="mb-1 flex items-center justify-between text-xs">
            <span className="text-emerald-400">Belief (Fraud)</span>
            <span className="font-mono text-muted-foreground">{bPct}%</span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-secondary">
            <div
              className="h-full rounded-full bg-emerald-500 transition-all duration-500"
              style={{ width: `${bPct}%` }}
            />
          </div>
        </div>
        <div>
          <div className="mb-1 flex items-center justify-between text-xs">
            <span className="text-red-400">Disbelief (Legit)</span>
            <span className="font-mono text-muted-foreground">{dPct}%</span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-secondary">
            <div
              className="h-full rounded-full bg-red-500 transition-all duration-500"
              style={{ width: `${dPct}%` }}
            />
          </div>
        </div>
        <div>
          <div className="mb-1 flex items-center justify-between text-xs">
            <span className="text-amber-400">Uncertainty</span>
            <span className="font-mono text-muted-foreground">{uPct}%</span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-secondary">
            <div
              className="h-full rounded-full bg-amber-500 transition-all duration-500"
              style={{ width: `${uPct}%` }}
            />
          </div>
        </div>
      </div>

      {/* Visual triangle representation */}
      <div className="flex items-center justify-center py-2">
        <svg width="140" height="120" viewBox="0 0 140 120">
          <polygon
            points="70,10 130,110 10,110"
            fill="none"
            stroke="#334155"
            strokeWidth="1"
          />
          {/* Point position based on barycentric coords */}
          <circle
            cx={70 + (disbelief - belief) * 50}
            cy={110 - uncertainty * 90}
            r="6"
            fill="#6366f1"
            stroke="#a5b4fc"
            strokeWidth="2"
          />
          {/* Labels */}
          <text x="70" y="7" textAnchor="middle" fill="#94a3b8" fontSize="9">
            Uncertainty
          </text>
          <text x="5" y="122" textAnchor="start" fill="#94a3b8" fontSize="9">
            Belief
          </text>
          <text x="135" y="122" textAnchor="end" fill="#94a3b8" fontSize="9">
            Disbelief
          </text>
        </svg>
      </div>
    </div>
  );
}
