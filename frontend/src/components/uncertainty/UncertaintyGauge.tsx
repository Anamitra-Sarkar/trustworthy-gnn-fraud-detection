"use client";

import {
  RadialBarChart,
  RadialBar,
  ResponsiveContainer,
  PolarAngleAxis,
} from "recharts";

interface UncertaintyGaugeProps {
  value: number;
  label?: string;
  size?: number;
}

export default function UncertaintyGauge({
  value,
  label = "Confidence",
  size = 160,
}: UncertaintyGaugeProps) {
  const percentage = Math.round(value * 100);
  const color =
    percentage >= 80
      ? "#22c55e"
      : percentage >= 60
        ? "#eab308"
        : percentage >= 40
          ? "#f97316"
          : "#ef4444";

  const data = [{ name: label, value: percentage, fill: color }];

  return (
    <div className="flex flex-col items-center">
      <div style={{ width: size, height: size }} className="relative">
        <ResponsiveContainer>
          <RadialBarChart
            cx="50%"
            cy="50%"
            innerRadius="75%"
            outerRadius="100%"
            data={data}
            startAngle={90}
            endAngle={-270}
            barSize={8}
          >
            <PolarAngleAxis
              type="number"
              domain={[0, 100]}
              angleAxisId={0}
              tick={false}
            />
            <RadialBar
              background={{ fill: "#1e293b" }}
              dataKey="value"
              cornerRadius={4}
              angleAxisId={0}
            />
          </RadialBarChart>
        </ResponsiveContainer>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-xl font-bold text-foreground">
            {percentage}%
          </span>
        </div>
      </div>
      <span className="mt-1 text-xs text-muted-foreground">{label}</span>
    </div>
  );
}
