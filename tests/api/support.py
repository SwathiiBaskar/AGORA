import asyncio
from collections.abc import Callable
from typing import Any

import httpx
from fastapi import FastAPI

LogEntries = Callable[[], list[dict[str, Any]]]


class AppClient:
    """Calls an ASGI application in-process, without a network or a server.

    Unhandled application exceptions are not re-raised into the test: the API is
    expected to turn them into a 500 response, and that is what is asserted.
    """

    def __init__(self, app: FastAPI) -> None:
        self.app = app

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        async def send() -> httpx.Response:
            transport = httpx.ASGITransport(app=self.app, raise_app_exceptions=False)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
                return await c.request(method, url, **kwargs)

        return asyncio.run(send())

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", url, **kwargs)
