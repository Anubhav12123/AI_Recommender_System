# services/api/app/main.py
from fastapi import FastAPI
from .routers import health, search, recommend, feedback, metrics

app = FastAPI(title="AI Reco System")

app.include_router(health.router)
app.include_router(search.router)
app.include_router(recommend.router)
app.include_router(feedback.router)
app.include_router(metrics.router)
