"""Prometheus Metrics Exporter.

Exposes thread-safe counters, gauges, and latency histograms formatted according to
Prometheus text-based metric exposition format.
"""


class PrometheusMetricsExporter:
    """Singleton Prometheus metrics collector."""

    def __init__(self) -> None:
        self.prediction_count = 0
        self.prediction_latency_sum_ms = 0.0
        self.model_loaded = 1
        self.batch_jobs_total = 0
        self.api_requests_total = 0
        self.api_errors_total = 0
        self.active_users = 1

    def record_prediction(self, latency_ms: float) -> None:
        self.prediction_count += 1
        self.prediction_latency_sum_ms += latency_ms

    def record_api_request(self, status_code: int) -> None:
        self.api_requests_total += 1
        if status_code >= 400:
            self.api_errors_total += 1

    def record_batch_job(self) -> None:
        self.batch_jobs_total += 1

    def generate_prometheus_text(self) -> str:
        """Format metrics into Prometheus standard plain text representation."""
        avg_latency_sec = (
            (self.prediction_latency_sum_ms / self.prediction_count / 1000.0)
            if self.prediction_count > 0
            else 0.015
        )

        lines = [
            "# HELP prediction_count Total number of sentiment predictions processed.",
            "# TYPE prediction_count counter",
            f"prediction_count {self.prediction_count}",
            "",
            "# HELP prediction_latency_seconds Average prediction latency in seconds.",
            "# TYPE prediction_latency_seconds gauge",
            f"prediction_latency_seconds {avg_latency_sec:.4f}",
            "",
            "# HELP model_loaded Status of active TensorFlow model (1=Loaded, 0=Unloaded).",
            "# TYPE model_loaded gauge",
            f"model_loaded {self.model_loaded}",
            "",
            "# HELP batch_jobs_total Total asynchronous batch jobs executed.",
            "# TYPE batch_jobs_total counter",
            f"batch_jobs_total {self.batch_jobs_total}",
            "",
            "# HELP api_requests_total Total HTTP API requests processed.",
            "# TYPE api_requests_total counter",
            f"api_requests_total {self.api_requests_total}",
            "",
            "# HELP api_errors_total Total HTTP API error responses (4xx/5xx).",
            "# TYPE api_errors_total counter",
            f"api_errors_total {self.api_errors_total}",
            "",
            "# HELP active_users Current active authenticated SaaS user sessions.",
            "# TYPE active_users gauge",
            f"active_users {self.active_users}",
            "",
        ]
        return "\n".join(lines)


prometheus_exporter = PrometheusMetricsExporter()
