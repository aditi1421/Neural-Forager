from fastapi.testclient import TestClient

from neural_forager.server import app


def test_dashboard_assets_and_health():
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert "Neural Forager" in client.get("/").text
        assert client.get("/assets/app.js").status_code == 200
        assert client.get("/assets/style.css").status_code == 200
        assert client.get("/health").json()["engine"] == "nengo"
        assert client.get("/assets/secret.txt").status_code == 422


def test_websocket_step_intervention_export_and_reset():
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "reset", "config": {"agent": "tabular", "steps": 60}})
        assert ws.receive_json()["data"]["world"]["step"] == 0
        ws.send_json({"type": "step", "count": 4})
        assert ws.receive_json()["data"]["world"]["step"] == 4
        ws.send_json({"type": "food"})
        assert ws.receive_json()["data"]["world"]["events"][-1]["kind"] == "food"
        ws.send_json({"type": "export"})
        export = ws.receive_json()
        assert export["type"] == "export" and len(export["data"]["history"]) == 4
        ws.send_json({"type": "step", "count": 500})
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "reset", "config": {"agent": "random", "seed": 12}})
        data = ws.receive_json()["data"]
        assert data["world"]["step"] == 0 and data["world"]["events"] == []


def test_connections_have_isolated_worlds():
    with TestClient(app) as client, client.websocket_connect("/ws") as a, client.websocket_connect("/ws") as b:
        for socket in [a, b]:
            socket.send_json({"type": "reset", "config": {"agent": "random"}})
            socket.receive_json()
        a.send_json({"type": "step", "count": 5})
        a.receive_json()
        b.send_json({"type": "export"})
        assert b.receive_json()["data"]["summary"]["steps"] == 0


def test_neural_step_delivers_real_activity_over_websocket():
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "reset", "config": {"agent": "neural", "steps": 30}})
        assert ws.receive_json()["data"]["brain"]["neurons"] == 1552
        ws.send_json({"type": "step"})
        response = ws.receive_json()
        assert response["type"] == "state"
        assert response["data"]["world"]["step"] == 1
        brain = response["data"]["brain"]
        assert brain["weight_change"] > 0
        assert max(max(row) for row in brain["activity"]) > 0
