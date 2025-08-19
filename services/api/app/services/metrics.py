from prometheus_client import Counter, Histogram

REQS = Counter("api_requests_total", "Total API requests", ["route"])
LAT  = Histogram("api_latency_seconds", "Latency per route", ["route"])
