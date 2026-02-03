"use client";

import { useState } from "react";

interface AnalysisFormProps {
  onSubmit: (ticker: string, forecastHorizon?: 7 | 30) => void;
  loading: boolean;
}

export default function AnalysisForm({ onSubmit, loading }: AnalysisFormProps) {
  const [ticker, setTicker] = useState("");
  const [forecastHorizon, setForecastHorizon] = useState<"" | "7" | "30">("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (ticker.trim()) {
      onSubmit(
        ticker.toUpperCase(),
        forecastHorizon ? (Number(forecastHorizon) as 7 | 30) : undefined,
      );
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label
          htmlFor="ticker"
          className="block text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-1"
        >
          Stock Ticker
        </label>
        <input
          type="text"
          id="ticker"
          value={ticker}
          onChange={(e) => setTicker(e.target.value.toUpperCase())}
          placeholder="e.g., AAPL, MSFT, BMW"
          className="w-full px-4 py-2 border border-zinc-300 dark:border-zinc-600 rounded-lg bg-white dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          disabled={loading}
          required
          maxLength={5}
        />
        <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-1">
          1-5 uppercase letters (e.g., AAPL, MSFT, BMW)
        </p>
      </div>

      <div>
        <label
          htmlFor="forecast"
          className="block text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-1"
        >
          Forecast Horizon
        </label>
        <select
          id="forecast"
          value={forecastHorizon}
          onChange={(e) =>
            setForecastHorizon(e.target.value as "" | "7" | "30")
          }
          className="w-full px-4 py-2 border border-zinc-300 dark:border-zinc-600 rounded-lg bg-white dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          disabled={loading}
        >
          <option value="">No forecast</option>
          <option value="7">7 days</option>
          <option value="30">30 days</option>
        </select>
      </div>

      <button
        type="submit"
        disabled={loading || !ticker.trim()}
        className="w-full bg-blue-600 text-white py-2.5 px-4 rounded-lg font-medium hover:bg-blue-700 disabled:bg-zinc-400 disabled:cursor-not-allowed transition-colors"
      >
        {loading ? "Analyzing..." : "Analyze"}
      </button>
    </form>
  );
}
