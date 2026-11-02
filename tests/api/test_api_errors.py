from typing import Any

import pytest
from fastapi import FastAPI, HTTPException

from civbox.api.app import normalise_openapi
from civbox.api.errors import MAX_VALIDATION_DETAILS
from civbox.core.errors import ConfigurationError

from .support import AppClient, LogEntries


def error_of(response: Any) -> dict[str, Any]:
    body = response.json()
    assert list(body) == ["error"]
    error: dict[str, Any] = body["error"]
    assert error["request_id"] == response.headers["x-request-id"]
    return error


def test_unknown_paths_use_the_error_envelope(client: AppClient) -> None:
    response = client.get("/nope")
    assert response.status_code == 404
    assert error_of(response) == {
        "code": "not_found",
        "message": "Not Found",
        "request_id": response.headers["x-request-id"],
    }


def test_wrong_methods_use_the_envelope_and_keep_the_allow_header(client: AppClient) -> None:
    response = client.post("/healthz")
    assert response.status_code == 405
    assert error_of(response)["code"] == "method_not_allowed"
    assert response.headers["allow"] == "GET"


def test_routes_can_raise_http_errors_with_a_safe_message_and_headers(
    app: FastAPI, client: AppClient
) -> None:
    @app.get("/conflict")
    async def conflict() -> None:
        raise HTTPException(409, "name already exists", headers={"X-Reason": "duplicate"})

    response = client.get("/conflict")
    assert response.status_code == 409
    error = error_of(response)
    assert (error["code"], error["message"]) == ("conflict", "name already exists")
    assert response.headers["x-reason"] == "duplicate"


def test_unmapped_status_codes_get_a_generic_code(app: FastAPI, client: AppClient) -> None:
    @app.get("/teapot")
    async def teapot() -> None:
        raise HTTPException(418)

    assert error_of(client.get("/teapot"))["code"] == "http_418"


def test_secrets_in_http_error_messages_are_redacted(app: FastAPI, client: AppClient) -> None:
    @app.get("/leaky")
    async def leaky() -> None:
        raise HTTPException(400, "cannot reach postgresql://app:hunter2@db/x")

    assert "hunter2" not in client.get("/leaky").text


def test_validation_errors_report_location_message_and_type(
    app: FastAPI, client: AppClient
) -> None:
    @app.get("/items")
    async def items(limit: int) -> dict[str, int]:
        return {"limit": limit}

    response = client.get("/items", params={"limit": "many"})
    assert response.status_code == 422
    error = error_of(response)
    assert error["code"] == "validation_error"
    assert error["message"] == "Request validation failed"
    assert error["details"] == [
        {
            "location": ["query", "limit"],
            "message": "Input should be a valid integer, unable to parse string as an integer",
            "type": "int_parsing",
        }
    ]


def test_validation_errors_never_echo_the_rejected_input(app: FastAPI, client: AppClient) -> None:
    @app.get("/items")
    async def items(limit: int) -> dict[str, int]:
        return {"limit": limit}

    sentinel = "sup3r-s3cret-input"
    assert sentinel not in client.get("/items", params={"limit": sentinel}).text


def test_validation_details_are_capped(app: FastAPI, client: AppClient) -> None:
    @app.post("/numbers")
    async def numbers(values: list[int]) -> None:
        return None

    response = client.post("/numbers", json=["x"] * (MAX_VALIDATION_DETAILS * 2))
    assert response.status_code == 422
    assert len(error_of(response)["details"]) == MAX_VALIDATION_DETAILS


def test_unexpected_exceptions_return_a_generic_500(
    app: FastAPI, client: AppClient, json_logs: LogEntries
) -> None:
    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("detail postgresql://app:hunter2@db/x")

    response = client.get("/boom")
    assert response.status_code == 500
    assert error_of(response) == {
        "code": "internal_error",
        "message": "An internal error occurred.",
        "request_id": response.headers["x-request-id"],
    }
    assert "detail" not in response.text
    assert "hunter2" not in response.text


def test_unexpected_exceptions_are_logged_with_redacted_detail_and_the_request_id(
    app: FastAPI, client: AppClient, json_logs: LogEntries
) -> None:
    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("detail postgresql://app:hunter2@db/x")

    response = client.get("/boom")
    failures = [e for e in json_logs() if e["level"] == "ERROR"]
    assert len(failures) == 1
    assert failures[0]["request_id"] == response.headers["x-request-id"]
    assert failures[0]["exception"]["type"] == "RuntimeError"
    assert "hunter2" not in str(json_logs())


def test_deliberate_application_errors_expose_only_their_code(
    app: FastAPI, client: AppClient
) -> None:
    @app.get("/misconfigured")
    async def misconfigured() -> None:
        raise ConfigurationError([("LOG_LEVEL", "bad value")])

    response = client.get("/misconfigured")
    assert response.status_code == 500
    error = error_of(response)
    assert error["code"] == "configuration_error"
    assert error["message"] == "An internal error occurred."
    assert "LOG_LEVEL" not in response.text


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/docs/oauth2-redirect"])
def test_interactive_documentation_pages_are_not_served(client: AppClient, path: str) -> None:
    assert client.get(path).status_code == 404


def test_openapi_document_is_served_and_describes_the_envelope(client: AppClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    document = response.json()
    assert {"/healthz", "/readyz"} <= set(document["paths"])
    assert {"ErrorResponse", "ErrorBody", "ErrorDetail"} <= set(document["components"]["schemas"])
    ok = document["paths"]["/healthz"]["get"]["responses"]
    assert ok["500"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/ErrorResponse"
    }


def test_openapi_documents_validation_errors_as_the_envelope(
    app: FastAPI, client: AppClient
) -> None:
    @app.get("/items")
    async def items(limit: int) -> dict[str, int]:
        return {"limit": limit}

    document = client.get("/openapi.json").json()
    responses = document["paths"]["/items"]["get"]["responses"]
    assert responses["422"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/ErrorResponse"
    }
    assert "HTTPValidationError" not in document["components"]["schemas"]
    assert "ValidationError" not in document["components"]["schemas"]


def test_openapi_normalisation_is_idempotent() -> None:
    schema: dict[str, Any] = {
        "paths": {
            "/x": {
                "parameters": [],
                "get": {"responses": {"422": {"description": "Validation Error"}}},
            }
        },
        "components": {"schemas": {"HTTPValidationError": {}, "ValidationError": {}, "Keep": {}}},
    }
    once = normalise_openapi(schema)
    snapshot = repr(once)
    assert repr(normalise_openapi(once)) == snapshot
    assert set(once["components"]["schemas"]) == {"Keep"}
