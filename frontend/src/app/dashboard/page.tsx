"use client";

import { useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import {
  Activity,
  AlertTriangle,
  ShieldCheck,
  TrendingUp,
  Clock,
} from "lucide-react";
import Header from "@/components/layout/Header";
import { cn } from "@/lib/utils";

const stats = [
  {
    label: "Total Analyses",
    value: "12,847",
    change: "+12.5%",
    icon: Activity,
    color: "from-primary to-indigo-400",
  },
  {
    label: "Flagged Transactions",
    value: "342",
    change: "+3.2%",
    icon: AlertTriangle,
    color: "from-amber-500 to-orange-400",
  },
  {
    label: "Pending Escalations",
    value: "28",
    change: "-8.1%",
    icon: ShieldCheck,
    color: "from-emerald-500 to-teal-400",
  },
  {
    label: "Model Accuracy",
    value: "97.3%",
    change: "+0.4%",
    icon: TrendingUp,
    color: "from-violet-500 to-purple-400",
  },
];

const modelPerformance = [
  { backbone: "GCN", f1: 0.91, auc: 0.95 },
  { backbone: "GAT", f1: 0.93, auc: 0.96 },
  { backbone: "GraphSAGE", f1: 0.94, auc: 0.97 },
  { backbone: "GIN", f1: 0.92, auc: 0.95 },
  { backbone: "GAT-v2", f1: 0.95, auc: 0.98 },
];

const recentActivity = [
  {
    id: "TXN-8834",
    action: "Flagged as suspicious",
    risk: "high",
    time: "2 min ago",
  },
  {
    id: "TXN-8831",
    action: "Compliance report generated",
    risk: "medium",
    time: "15 min ago",
  },
  {
    id: "TXN-8829",
    action: "Escalation resolved",
    risk: "low",
    time: "32 min ago",
  },
  {
    id: "TXN-8825",
    action: "New analysis completed",
    risk: "critical",
    time: "1 hr ago",
  },
  {
    id: "TXN-8820",
    action: "Batch inference finished",
    risk: "low",
    time: "2 hr ago",
  },
];

const riskColors: Record<string, string> = {
  critical: "text-red-400 bg-red-400/10",
  high: "text-orange-400 bg-orange-400/10",
  medium: "text-amber-400 bg-amber-400/10",
  low: "text-emerald-400 bg-emerald-400/10",
};

const CHART_COLORS = ["#6366f1", "#8b5cf6"];

export default function DashboardPage() {
  const [activeMetric, setActiveMetric] = useState<"f1" | "auc">("auc");

  return (
    <>
      <Header title="Dashboard" />
      <div className="space-y-6 p-6">
        {/* Stat Cards */}
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {stats.map((stat) => (
            <div key={stat.label} className="gradient-border p-5">
              <div className="flex items-start justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">{stat.label}</p>
                  <p className="mt-1 text-2xl font-bold text-foreground">
                    {stat.value}
                  </p>
                </div>
                <div
                  className={cn(
                    "flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br",
                    stat.color
                  )}
                >
                  <stat.icon className="h-5 w-5 text-white" />
                </div>
              </div>
              <p className="mt-3 text-xs text-muted-foreground">
                <span
                  className={cn(
                    "font-medium",
                    stat.change.startsWith("+")
                      ? "text-emerald-400"
                      : "text-red-400"
                  )}
                >
                  {stat.change}
                </span>{" "}
                vs last month
              </p>
            </div>
          ))}
        </div>

        <div className="grid gap-6 lg:grid-cols-5">
          {/* Model Performance Chart */}
          <div className="glass rounded-xl p-6 lg:col-span-3">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-base font-semibold text-foreground">
                Model Performance by Backbone
              </h2>
              <div className="flex gap-1 rounded-lg bg-secondary p-1">
                <button
                  onClick={() => setActiveMetric("auc")}
                  className={cn(
                    "rounded-md px-3 py-1 text-xs font-medium transition-colors",
                    activeMetric === "auc"
                      ? "bg-primary text-white"
                      : "text-muted-foreground hover:text-foreground"
                  )}
                >
                  AUC
                </button>
                <button
                  onClick={() => setActiveMetric("f1")}
                  className={cn(
                    "rounded-md px-3 py-1 text-xs font-medium transition-colors",
                    activeMetric === "f1"
                      ? "bg-primary text-white"
                      : "text-muted-foreground hover:text-foreground"
                  )}
                >
                  F1
                </button>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart
                data={modelPerformance}
                margin={{ top: 5, right: 5, bottom: 5, left: -10 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis
                  dataKey="backbone"
                  stroke="#64748b"
                  fontSize={12}
                  tickLine={false}
                />
                <YAxis
                  domain={[0.85, 1]}
                  stroke="#64748b"
                  fontSize={12}
                  tickLine={false}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#0f172a",
                    border: "1px solid #1e293b",
                    borderRadius: "8px",
                    color: "#e2e8f0",
                    fontSize: "12px",
                  }}
                />
                <Bar
                  dataKey={activeMetric}
                  radius={[4, 4, 0, 0]}
                  maxBarSize={40}
                >
                  {modelPerformance.map((_, index) => (
                    <Cell
                      key={index}
                      fill={CHART_COLORS[index % CHART_COLORS.length]}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Recent Activity */}
          <div className="glass rounded-xl p-6 lg:col-span-2">
            <h2 className="mb-4 text-base font-semibold text-foreground">
              Recent Activity
            </h2>
            <div className="space-y-3">
              {recentActivity.map((entry) => (
                <div
                  key={entry.id}
                  className="flex items-start gap-3 rounded-lg p-3 transition-colors hover:bg-secondary/50"
                >
                  <div className="mt-0.5">
                    <Clock className="h-4 w-4 text-muted-foreground" />
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-foreground">
                        {entry.id}
                      </span>
                      <span
                        className={cn(
                          "rounded-full px-2 py-0.5 text-[10px] font-medium capitalize",
                          riskColors[entry.risk]
                        )}
                      >
                        {entry.risk}
                      </span>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {entry.action}
                    </p>
                    <p className="mt-0.5 text-[10px] text-muted-foreground/60">
                      {entry.time}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
