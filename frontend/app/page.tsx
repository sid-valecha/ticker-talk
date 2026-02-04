"use client";

import { useState, useEffect } from "react";
import {
  analyzeStock,
  getAvailableTickers,
  type AnalysisResponse,
} from "@/app/lib/api";
import AnalysisForm from "@/app/components/AnalysisForm";
import AnalysisResults from "@/app/components/AnalysisResults";

export default function Home() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<AnalysisResponse | null>(null);
  const [availableTickers, setAvailableTickers] = useState<string[]>([]);

  useEffect(() => {
    getAvailableTickers()
      .then(setAvailableTickers)
      .catch(() => setAvailableTickers([]));
  }, []);

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
    <div className="min-h-screen bg-zinc-50 dark:bg-black">
      <div className="max-w-5xl mx-auto px-4 py-8">
        {/* Header */}
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold text-zinc-900 dark:text-zinc-50 mb-2">
            Ticker Talk
          </h1>
          <p className="text-zinc-600 dark:text-zinc-400">
            Stock analysis with technical indicators and forecasting
          </p>
        </header>

        {/* Input */}
        <div className="max-w-md mx-auto mb-8">
          <div className="rounded-lg border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 p-6">
            <AnalysisForm
              onSubmit={handleSubmit}
              loading={loading}
              availableTickers={availableTickers}
            />
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="max-w-md mx-auto mb-8">
            <div className="rounded-lg border border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-950 p-4">
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
