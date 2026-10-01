import asyncio
import json
from importlib.resources import files
from threading import Lock
from typing import Literal

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .config import ExperimentConfig
from .experiment import Experiment


class Command(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["reset", "step", "route", "food", "export"]
    config: ExperimentConfig = Field(default_factory=ExperimentConfig)
    count: int = Field(default=1, ge=1, le=10)


class Session:
    def __init__(self):
        self.lock = Lock()
        self.experiment = None

    def command(self, command: Command):
        with self.lock:
            if command.type == "reset":
                candidate = Experiment(command.config)
                if self.experiment:
                    self.experiment.close()
                self.experiment = candidate
            if self.experiment is None:
                raise ValueError("Reset the experiment first")
            if command.type == "step":
                for _ in range(command.count):
                    self.experiment.step()
            elif command.type in ("route", "food"):
                self.experiment.intervene(command.type)
            elif command.type == "export":
                return {"type": "export", "data": {
                    "config": self.experiment.config.model_dump(),
                    "summary": self.experiment.summary(), "trials": self.experiment.trials,
                    "history": self.experiment.history}}
            return {"type": "state", "data": self.experiment.snapshot()}

    def close(self):
        with self.lock:
            if self.experiment:
                self.experiment.close()


app = FastAPI(title="Neural Forager")


@app.get("/", response_class=HTMLResponse)
def index():
    return files("neural_forager").joinpath("static/index.html").read_text()


@app.get("/assets/{name}")
def asset(name: Literal["app.js", "style.css"]):
    return Response(files("neural_forager").joinpath("static", name).read_text(),
                    media_type="application/javascript" if name.endswith("js") else "text/css")


@app.get("/health")
def health():
    return {"status": "ok", "engine": "nengo"}


@app.websocket("/ws")
async def websocket(ws: WebSocket):
    # A localhost dashboard should not accept cross-site drive-by simulation jobs.
    origin = ws.headers.get("origin")
    if origin and origin not in {"http://" + ws.headers.get("host", ""),
                                 "https://" + ws.headers.get("host", "")}:
        await ws.close(code=1008)
        return
    await ws.accept()
    session = Session()
    try:
        while True:
            payload = await ws.receive_text()
            try:
                command = Command.model_validate_json(payload)
                result = await asyncio.to_thread(session.command, command)
                await ws.send_json(result)
            except (ValidationError, ValueError) as exc:
                await ws.send_json({"type": "error", "message": str(exc)})
    except WebSocketDisconnect:
        pass
    finally:
        await asyncio.to_thread(session.close)
