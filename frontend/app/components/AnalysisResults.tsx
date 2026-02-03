import type { AnalysisResponse } from "@/app/lib/api";

interface AnalysisResultsProps {
  data: AnalysisResponse;
}

export default function AnalysisResults({ data }: AnalysisResultsProps) {
  const { metadata, indicators, forecast, backtest, plots } = data;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="rounded-lg border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 p-6">
        <h2 className="text-2xl font-bold text-zinc-900 dark:text-zinc-50 mb-2">
          {metadata.ticker}
        </h2>
        <div className="text-sm text-zinc-600 dark:text-zinc-400 space-y-1">
          <p>
            Period: {metadata.min_date} to {metadata.max_date}
          </p>
          <p>
            {metadata.row_count} data points &middot; Source: {metadata.source}{" "}
            &middot; Cache: {metadata.cache_hit ? "hit" : "miss"}
          </p>
        </div>
      </div>

      {/* Key Indicators */}
      {indicators && (
        <div className="rounded-lg border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 p-6">
          <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50 mb-4">
            Key Indicators
          </h3>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
            <Metric
              label="Latest Price"
              value={`$${indicators.latest_price.toFixed(2)}`}
            />
            <Metric
              label="20-Day MA"
              value={`$${indicators.ma_20_latest.toFixed(2)}`}
            />
            <Metric
              label="RSI (14)"
              value={indicators.rsi_14_latest.toFixed(1)}
              note={rsiNote(indicators.rsi_14_latest)}
            />
            <Metric
              label="Volatility (20d)"
              value={`${(indicators.volatility_20_latest * 100).toFixed(2)}%`}
            />
            <Metric
              label="Latest Return"
              value={`${(indicators.latest_return * 100).toFixed(2)}%`}
            />
            <Metric
              label="30d Avg Return"
              value={`${(indicators.avg_return_30d * 100).toFixed(3)}%`}
            />
            <Metric
              label="30d Avg Volatility"
              value={`${(indicators.avg_volatility_30d * 100).toFixed(2)}%`}
            />
          </div>
        </div>
      )}

      {/* Explanation */}
      {data.explanation && (
        <div className="rounded-lg border border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-950 p-6">
          <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50 mb-3">
            Analysis Summary
          </h3>
          <div className="text-sm text-zinc-700 dark:text-zinc-300 whitespace-pre-line leading-relaxed">
            {data.explanation}
          </div>
        </div>
      )}

      {/* Forecast */}
      {forecast && (
        <div className="rounded-lg border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 p-6">
          <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50 mb-4">
            {forecast.horizon}-Day Forecast
          </h3>
          <div className="space-y-2 text-sm text-zinc-700 dark:text-zinc-300">
            <p>
              <span className="font-medium">Trend:</span>{" "}
              <span
                className={
                  forecast.trend === "upward"
                    ? "text-green-600 dark:text-green-400"
                    : forecast.trend === "downward"
                      ? "text-red-600 dark:text-red-400"
                      : "text-zinc-600 dark:text-zinc-400"
                }
              >
                {forecast.trend}
              </span>
            </p>
            <p>
              <span className="font-medium">Predicted End Price:</span> $
              {forecast.forecast[forecast.forecast.length - 1].toFixed(2)}
            </p>
            <p>
              <span className="font-medium">95% Confidence Interval:</span> $
              {forecast.lower_ci[forecast.lower_ci.length - 1].toFixed(2)} - $
              {forecast.upper_ci[forecast.upper_ci.length - 1].toFixed(2)}
            </p>
          </div>

          {backtest && (
            <div className="mt-4 pt-4 border-t border-zinc-200 dark:border-zinc-700">
              <p className="text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-2">
                Model Accuracy (Backtest)
              </p>
              <div className="grid grid-cols-3 gap-4">
                <Metric
                  label="MAE"
                  value={`$${backtest.mae.toFixed(2)}`}
                  small
                />
                <Metric
                  label="RMSE"
                  value={`$${backtest.rmse.toFixed(2)}`}
                  small
                />
                <Metric
                  label="MAPE"
                  value={`${backtest.mape.toFixed(1)}%`}
                  small
                />
              </div>
            </div>
          )}
        </div>
      )}

      {/* Charts */}
      {plots && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50">
            Charts
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {plots.price_and_ma && (
              <ChartImage
                title="Price & Moving Average"
                base64={plots.price_and_ma}
              />
            )}
            {plots.returns_volatility && (
              <ChartImage
                title="Returns & Volatility"
                base64={plots.returns_volatility}
              />
            )}
            {plots.rsi && <ChartImage title="RSI" base64={plots.rsi} />}
            {plots.forecast && (
              <ChartImage title="Forecast" base64={plots.forecast} />
            )}
          </div>
        </div>
      )}

      {/* Disclaimer */}
      <div className="rounded-lg border border-yellow-200 dark:border-yellow-800 bg-yellow-50 dark:bg-yellow-950 p-4 text-xs text-zinc-600 dark:text-zinc-400">
        <p className="font-medium mb-1">Disclaimer</p>
        <p>
          This analysis is for educational purposes only. Not financial advice.
          Past performance does not guarantee future results. ARIMA forecasts are
          statistical predictions with uncertainty.
        </p>
      </div>
    </div>
  );
}

function rsiNote(rsi: number): string | undefined {
  if (rsi > 70) return "overbought";
  if (rsi < 30) return "oversold";
  return undefined;
}

function Metric({
  label,
  value,
  note,
  small,
}: {
  label: string;
  value: string;
  note?: string;
  small?: boolean;
}) {
  return (
    <div>
      <p
        className={`text-zinc-500 dark:text-zinc-400 ${small ? "text-xs" : "text-sm"}`}
      >
        {label}
      </p>
      <p
        className={`font-semibold text-zinc-900 dark:text-zinc-100 ${small ? "text-sm" : "text-lg"}`}
      >
        {value}
      </p>
      {note && (
        <p className="text-xs text-amber-600 dark:text-amber-400">{note}</p>
      )}
    </div>
  );
}

function ChartImage({ title, base64 }: { title: string; base64: string }) {
  return (
    <div className="rounded-lg border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 p-4">
      <h4 className="font-medium text-zinc-900 dark:text-zinc-100 mb-2 text-sm">
        {title}
      </h4>
      <img
        src={`data:image/png;base64,${base64}`}
        alt={title}
        className="w-full rounded"
      />
    </div>
  );
}
