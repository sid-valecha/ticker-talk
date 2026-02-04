"use client";

import { useState } from "react";

interface AnalysisFormProps {
  onSubmit: (ticker: string, forecastHorizon?: 7 | 30) => void;
  onParseIntent: (query: string) => Promise<{
    ticker?: string;
    forecast_horizon?: 7 | 30 | null;
    error?: string;
  }>;
  loading: boolean;
  exampleQueries: string[];
}

export default function AnalysisForm({
  onSubmit,
  onParseIntent,
  loading,
  exampleQueries,
}: AnalysisFormProps) {
  const [query, setQuery] = useState("");
  const [parsing, setParsing] = useState(false);
  const [parsedIntent, setParsedIntent] = useState<{
    ticker: string;
    forecast_horizon?: 7 | 30 | null;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const submitQuery = async (queryText: string) => {
    if (!queryText.trim() || parsing || loading) return;

    setError(null);
    setParsing(true);

    try {
      const result = await onParseIntent(queryText);

      if (result.error) {
        setError(result.error);
        setParsing(false);
        return;
      }

      if (!result.ticker) {
        setError("Could not identify ticker symbol");
        setParsing(false);
        return;
      }

      setParsedIntent({
        ticker: result.ticker,
        forecast_horizon: result.forecast_horizon || undefined,
      });

      setParsing(false);

      // Automatically submit analysis
      onSubmit(result.ticker, result.forecast_horizon || undefined);
    } catch (err) {
      setError("Failed to process query. Please try again.");
      setParsing(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    await submitQuery(query);
  };

  const handleExampleClick = (example: string) => {
    setQuery(example);
    setParsedIntent(null);
    setError(null);
    void submitQuery(example);
  };

  return (
    <div className="space-y-4">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label
            htmlFor="query"
            className="block text-sm font-medium mb-2 text-gray-700 dark:text-gray-300"
          >
            Ask me about any stock
          </label>
          <input
            type="text"
            id="query"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Try: 'forecast AAPL for 30 days'"
            className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-zinc-800 text-gray-900 dark:text-gray-100"
            disabled={loading || parsing}
            required
          />
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            You can use company names or ticker symbols
          </p>
        </div>

        {parsedIntent && !error && (
          <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-3">
            <p className="text-sm text-green-800 dark:text-green-200">
              Analyzing <span className="font-semibold">{parsedIntent.ticker}</span>
              {parsedIntent.forecast_horizon && (
                <span>
                  {" "}
                  with{" "}
                  <span className="font-semibold">
                    {parsedIntent.forecast_horizon}-day forecast
                  </span>
                </span>
              )}
            </p>
          </div>
        )}

        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-3">
            <p className="text-sm text-red-800 dark:text-red-200">{error}</p>
          </div>
        )}

        <button
          type="submit"
          disabled={loading || parsing || !query.trim()}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white py-3 px-4 rounded-lg disabled:bg-gray-400 disabled:cursor-not-allowed transition font-medium"
        >
          {parsing
            ? "Understanding query..."
            : loading
              ? "Analyzing & Generating Insights..."
              : "Analyze"}
        </button>
      </form>

      {exampleQueries.length > 0 && !loading && !parsing && (
        <div>
          <p className="text-xs text-gray-600 dark:text-gray-400 mb-2">
            Try these examples:
          </p>
          <div className="flex flex-wrap gap-2">
            {exampleQueries.slice(0, 3).map((example, idx) => (
              <button
                key={idx}
                onClick={() => handleExampleClick(example)}
                className="text-xs px-3 py-1.5 bg-gray-100 dark:bg-zinc-800 hover:bg-gray-200 dark:hover:bg-zinc-700 text-gray-700 dark:text-gray-300 rounded-full transition"
              >
                {example}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
