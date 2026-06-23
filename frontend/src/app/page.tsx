"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { Network, BarChart3, Shield, ArrowRight } from "lucide-react";

const features = [
  {
    icon: Network,
    title: "Graph Intelligence",
    description:
      "Leverage Graph Neural Networks to analyze complex transaction networks, uncovering hidden patterns and suspicious clusters that traditional methods miss.",
  },
  {
    icon: BarChart3,
    title: "Uncertainty Quantification",
    description:
      "Go beyond point predictions with MC Dropout, Conformal Prediction, and Evidential Deep Learning to quantify model confidence on every decision.",
  },
  {
    icon: Shield,
    title: "Compliance Automation",
    description:
      "Automated risk assessment reports, escalation workflows, and audit trails designed for regulatory compliance in financial institutions.",
  },
];

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.15, delayChildren: 0.3 },
  },
};

const item = {
  hidden: { opacity: 0, y: 24 },
  show: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.5, ease: "easeOut" as const },
  },
};

export default function LandingPage() {
  return (
    <div className="relative flex min-h-screen flex-col overflow-hidden">
      {/* Background gradient */}
      <div className="pointer-events-none fixed inset-0 -z-10">
        <div className="absolute inset-0 bg-gradient-to-br from-[#030712] via-[#1e1b4b] to-[#030712]" />
        <div className="absolute left-1/4 top-1/4 h-[500px] w-[500px] rounded-full bg-primary/5 blur-[120px]" />
        <div className="absolute bottom-1/4 right-1/4 h-[400px] w-[400px] rounded-full bg-accent/5 blur-[120px]" />
      </div>

      {/* Navbar */}
      <nav className="relative z-10 flex items-center justify-between px-8 py-6">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-primary to-accent">
            <Network className="h-5 w-5 text-white" />
          </div>
          <span className="text-xl font-bold tracking-tight">TrustGraph</span>
        </div>
        <Link
          href="/login"
          className="rounded-lg border border-border bg-secondary/50 px-5 py-2 text-sm font-medium text-foreground transition-all hover:border-primary/50 hover:bg-primary/10"
        >
          Sign In
        </Link>
      </nav>

      {/* Hero */}
      <main className="relative z-10 flex flex-1 flex-col items-center justify-center px-6 py-20">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: "easeOut" }}
          className="text-center"
        >
          <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-primary/20 bg-primary/5 px-4 py-1.5 text-xs font-medium text-primary">
            <span className="h-1.5 w-1.5 rounded-full bg-primary" />
            Financial Fraud Detection Platform
          </div>

          <h1 className="mx-auto max-w-4xl text-5xl font-bold leading-tight tracking-tight text-foreground sm:text-6xl lg:text-7xl">
            Trust
            <span className="bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
              Graph
            </span>
          </h1>

          <p className="mx-auto mt-6 max-w-2xl text-lg leading-relaxed text-muted-foreground">
            Trustworthy Graph Neural Networks for Financial Fraud Detection.
            Combining graph intelligence, uncertainty quantification, and
            compliance automation in one platform.
          </p>

          <div className="mt-10 flex items-center justify-center gap-4">
            <Link
              href="/login"
              className="group inline-flex items-center gap-2 rounded-lg bg-gradient-to-r from-primary to-accent px-6 py-3 text-sm font-semibold text-white shadow-lg shadow-primary/25 transition-all hover:shadow-xl hover:shadow-primary/30"
            >
              Get Started
              <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
            </Link>
            <Link
              href="/dashboard"
              className="rounded-lg border border-border px-6 py-3 text-sm font-medium text-muted-foreground transition-all hover:border-primary/30 hover:text-foreground"
            >
              View Demo
            </Link>
          </div>
        </motion.div>

        {/* Feature Cards */}
        <motion.div
          variants={container}
          initial="hidden"
          animate="show"
          className="mx-auto mt-24 grid max-w-5xl gap-6 sm:grid-cols-3"
        >
          {features.map((feature) => (
            <motion.div key={feature.title} variants={item}>
              <div className="glass group relative flex h-full flex-col rounded-xl p-6 transition-all duration-300 hover:border-primary/30">
                <div className="mb-4 flex h-11 w-11 items-center justify-center rounded-lg bg-primary/10 text-primary transition-colors group-hover:bg-primary/20">
                  <feature.icon className="h-5 w-5" />
                </div>
                <h3 className="mb-2 text-base font-semibold text-foreground">
                  {feature.title}
                </h3>
                <p className="text-sm leading-relaxed text-muted-foreground">
                  {feature.description}
                </p>
              </div>
            </motion.div>
          ))}
        </motion.div>
      </main>

      {/* Footer */}
      <footer className="relative z-10 border-t border-border px-8 py-6">
        <p className="text-center text-xs text-muted-foreground">
          TrustGraph &mdash; Trustworthy GNN Financial Fraud Detection System
        </p>
      </footer>
    </div>
  );
}
