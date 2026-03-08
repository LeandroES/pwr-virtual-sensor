from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.runs import router as runs_router
from app.api.sensor import router as sensor_router
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    yield


app = FastAPI(
    title="PWR Digital Twin API",
    description="Multiphysics digital twin for a Pressurized Water Reactor",
    version="0.1.0",
    debug=settings.debug,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    # Explicit list from settings instead of wildcard "*" so credentials
    # (cookies, auth headers) are supported and browsers don't reject CORS.
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(runs_router)
app.include_router(sensor_router)


@app.get("/health", tags=["ops"])
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "pwr-twin-api"}
