"use client";

import { useState, useEffect } from "react";
import {
  analyzeStock,
  getAvailableTickers,
  getExampleQueries,
  parseIntent,
  type AnalysisResponse,
} from "@/app/lib/api";
import AnalysisForm from "@/app/components/AnalysisForm";
import AnalysisResults from "@/app/components/AnalysisResults";

export default function Home() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<AnalysisResponse | null>(null);
  const [exampleQueries, setExampleQueries] = useState<string[]>([]);
  const [availableTickers, setAvailableTickers] = useState<string[]>([]);

  useEffect(() => {
    getExampleQueries()
      .then(setExampleQueries)
      .catch(() =>
        setExampleQueries([
          "forecast AAPL for 30 days",
          "show me Tesla",
          "analyze Microsoft with 7-day forecast",
        ]),
      );
  }, []);

  useEffect(() => {
    getAvailableTickers()
      .then(setAvailableTickers)
      .catch(() => setAvailableTickers([]));
  }, []);

  const handleParseIntent = async (query: string) => {
    return await parseIntent(query);
  };

  const handleSubmit = async (ticker: string, forecastHorizon?: 7 | 30) => {
    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const data = await analyzeStock({
        ticker,
        forecast_horizon: forecastHorizon,
      });
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen relative">
      <div
        className="pointer-events-none absolute inset-0 overflow-hidden"
        aria-hidden="true"
      >
        <div className="absolute -top-24 -left-16 h-72 w-72 rounded-full bg-blue-200/40 blur-3xl dark:bg-blue-500/20" />
        <div className="absolute top-6 right-0 h-64 w-64 rounded-full bg-cyan-200/40 blur-3xl dark:bg-cyan-500/20" />
      </div>
      <div className="relative max-w-5xl mx-auto px-4 py-10">
        {/* Header */}
        <header className="text-center mb-8">
          <h1 className="text-4xl sm:text-5xl font-bold text-zinc-900 dark:text-zinc-50 mb-2 tracking-tight">
            Ticker Talk
          </h1>
          <p className="text-zinc-600 dark:text-zinc-400">
            Ask me about any stock in natural language
          </p>
        </header>

        {/* Input */}
        <div className="max-w-md mx-auto mb-8">
          <div className="rounded-xl border border-zinc-200/80 dark:border-zinc-700/80 bg-white/80 dark:bg-zinc-900/70 p-6 shadow-sm backdrop-blur">
            <AnalysisForm
              onSubmit={handleSubmit}
              onParseIntent={handleParseIntent}
              loading={loading}
              exampleQueries={exampleQueries}
              availableTickers={availableTickers}
            />
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="max-w-md mx-auto mb-8">
            <div className="rounded-xl border border-red-200 dark:border-red-800 bg-red-50/90 dark:bg-red-950/70 p-4 shadow-sm backdrop-blur">
              <p className="text-sm text-red-800 dark:text-red-200">
                <span className="font-medium">Error: </span>
                {error}
              </p>
            </div>
          </div>
        )}

        {/* Results */}
        {results && <AnalysisResults data={results} />}
      </div>
    </div>
  );
}
