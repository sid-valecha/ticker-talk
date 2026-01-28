"use client";

import { useEffect, useState } from "react";

type HealthStatus = {
  status: string;
  environment: string;
} | null;

export default function Home() {
  const [health, setHealth] = useState<HealthStatus>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await fetch("http://localhost:8000/health");
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();
        setHealth(data);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Connection failed");
        setHealth(null);
      } finally {
        setLoading(false);
      }
    };

    checkHealth();
  }, []);

  return (
    <div className="flex min-h-screen items-center justify-center bg-zinc-50 dark:bg-black">
      <main className="flex flex-col items-center gap-8 p-8">
        <h1 className="text-4xl font-bold text-zinc-900 dark:text-zinc-50">
          Ticker-Talk
        </h1>
        <p className="text-lg text-zinc-600 dark:text-zinc-400">
          Natural-language stock analysis
        </p>

        <div className="mt-8 rounded-lg border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
          <h2 className="mb-4 text-lg font-semibold text-zinc-900 dark:text-zinc-50">
            Backend Status
          </h2>

          {loading && (
            <div className="flex items-center gap-2 text-zinc-500">
              <div className="h-3 w-3 animate-pulse rounded-full bg-zinc-400" />
              <span>Checking connection...</span>
            </div>
          )}

          {!loading && health && (
            <div className="flex items-center gap-2 text-green-600 dark:text-green-400">
              <div className="h-3 w-3 rounded-full bg-green-500" />
              <span>Connected ({health.environment})</span>
            </div>
          )}

          {!loading && error && (
            <div className="flex items-center gap-2 text-red-600 dark:text-red-400">
              <div className="h-3 w-3 rounded-full bg-red-500" />
              <span>Disconnected: {error}</span>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
