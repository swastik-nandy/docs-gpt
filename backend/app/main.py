from __future__ import annotations

from fastapi import FastAPI

from app.api.v1.router import api_router
from app.core.lifecycle import app_lifespan
from app.core.settings import get_settings


#------------------------APPLICATION-------------------------

def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        lifespan=app_lifespan,
    )

    app.include_router(
        api_router,
        prefix=settings.api_v1_prefix,
    )

    @app.get(
        "/health",
        tags=["health"],
    )
    async def health() -> dict[str, str]:
        event_bus = (
            app.state.redis_event_bus
        )

        try:
            redis_ok = (
                await event_bus.ping()
            )

        except Exception:
            redis_ok = False

        return {
            "status": (
                "ok"
                if redis_ok
                else "degraded"
            ),
            "redis": (
                "ok"
                if redis_ok
                else "unavailable"
            ),
            "chat": "ready",
        }

    return app


app = create_app()
