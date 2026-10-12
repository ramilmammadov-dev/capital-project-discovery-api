"""ASGI entry point: ``uvicorn capex.api.main:app``."""

from capex.api.app import create_app

app = create_app()
