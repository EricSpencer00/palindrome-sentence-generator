from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.v2 import RETIREMENT_MESSAGE, router


def test_standalone_v2_router_cannot_revive_legacy_output() -> None:
    app = FastAPI()
    app.include_router(router)
    response = TestClient(app).get("/api/v2/paragraph?sentences=2&source=catalogue")
    assert response.status_code == 503
    assert response.json()["detail"] == RETIREMENT_MESSAGE
