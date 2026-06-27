"use client";

import { useEffect, useState } from "react";
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
  BarChart3,
} from "lucide-react";
import Header from "@/components/layout/Header";
import { cn } from "@/lib/utils";
import { api } from "@/lib/api";
import { useAuth } from "@/hooks/useAuth";

type ModelSummary = {
  total_models: number;
  calibration_entries: number;
  best_model?: {
    model_name: string;
    backbone: string;
    f1_macro: number;
    f1_fraud: number;
    auc: number;
  } | null;
  model_performance: Array<{
    backbone: string;
    f1: number;
    auc: number;
  }>;
};

type StatCard = {
  label: string;
  value: string;
  change: string;
  icon: typeof Activity;
  color: string;
  tone?: "positive" | "neutral" | "negative";
};

const CHART_COLORS = ["#6366f1", "#8b5cf6", "#22c55e", "#f59e0b"];

type RecentActivity = {
  id: string;
  action: string;
  time: string;
};

export default function DashboardPage() {
  const { user, loading: authLoading } = useAuth();
  const [summary, setSummary] = useState<ModelSummary | null>(null);
  const [activeMetric, setActiveMetric] = useState<"f1" | "auc">("auc");

  useEffect(() => {
    if (authLoading || !user) {
      return;
    }

    let alive = true;
    (async () => {
      try {
        const data = (await api.getModelSummary()) as ModelSummary;
        if (alive) {
          setSummary(data);
        }
      } catch {
        if (alive) {
          setSummary(null);
        }
      }
    })();

    return () => {
      alive = false;
    };
  }, [authLoading, user]);

  const modelPerformance = summary?.model_performance?.length ? summary.model_performance : [];

  const stats: StatCard[] = summary
    ? [
        {
          label: "Trained Models",
          value: String(summary.total_models),
          change: "Live",
          icon: Activity,
          color: "from-primary to-indigo-400",
          tone: "neutral",
        },
        {
          label: "Best Fraud F1",
          value: summary.best_model ? summary.best_model.f1_fraud.toFixed(2) : "0.00",
          change: summary.best_model?.model_name ?? "No model",
          icon: TrendingUp,
          color: "from-emerald-500 to-teal-400",
          tone: "positive",
        },
        {
          label: "Best AUC",
          value: summary.best_model ? summary.best_model.auc.toFixed(2) : "0.00",
          change: summary.best_model?.backbone ?? "N/A",
          icon: ShieldCheck,
          color: "from-violet-500 to-purple-400",
          tone: "neutral",
        },
        {
          label: "Calibration Sets",
          value: String(summary.calibration_entries),
          change: "Conformal",
          icon: AlertTriangle,
          color: "from-amber-500 to-orange-400",
          tone: "neutral",
        },
      ]
    : [];

  const recentActivity: RecentActivity[] = modelPerformance.map((m, i) => ({
    id: `MODEL-${String(i + 1).padStart(3, "0")}`,
    action: `${m.backbone} — F1=${m.f1?.toFixed(3) ?? "N/A"} AUC=${m.auc?.toFixed(3) ?? "N/A"}`,
    time: "Live",
  }));

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
                    stat.tone === "positive"
                      ? "text-emerald-400"
                      : stat.tone === "negative"
                        ? "text-red-400"
                        : "text-slate-400"
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
                  domain={[0, 1]}
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
                  {recentActivity.length > 0 ? recentActivity.map((entry) => (
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
                        </div>
                        <p className="text-xs text-muted-foreground">
                          {entry.action}
                        </p>
                        <p className="mt-0.5 text-[10px] text-muted-foreground/60">
                          {entry.time}
                        </p>
                      </div>
                    </div>
                  )) : (
                    <div className="flex flex-col items-center justify-center py-8 text-center">
                      <Clock className="mb-2 h-8 w-8 text-muted-foreground/40" />
                      <p className="text-sm text-muted-foreground">No activity yet</p>
                    </div>
                  )}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
