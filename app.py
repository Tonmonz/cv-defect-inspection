"""Visual-inspection demo: a small web server (FastAPI) for the page in web/ and its JSON API.

    python app.py        # opens http://127.0.0.1:7861 in your browser (or double-click start_demo.bat)

Every model trained with `python -m src.train` (in artifacts/) appears in the product list.
API docs (Swagger) while it runs: http://127.0.0.1:7861/api/docs
"""
from __future__ import annotations

import argparse
import base64
import io
import socket
import threading
import time
import webbrowser
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from src.patchcore import FeatureExtractor, PatchCore, load_rgb, view
from src.paths import ARTIFACTS_DIR, REPORTS_DIR, ROOT
from src.plots import heat_layer
from src.regions import defect_regions

WEB_DIR = ROOT / "web"
SHOW = 448  # size of the pictures sent to the page


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def data_url(image: Image.Image, fmt: str) -> str:
    """Picture -> text the browser can show directly (no temporary files)."""
    buffer = io.BytesIO()
    image.save(buffer, fmt, **({"quality": 90} if fmt == "JPEG" else {}))
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buffer.getvalue()).decode()


class Inspector:
    """Loads the pretrained network once, and every trained model in artifacts/."""

    def __init__(self):
        self.extractor = FeatureExtractor()
        self.models = {}
        for meta_path in sorted(ARTIFACTS_DIR.glob("*/meta.json")):
            model, meta = PatchCore.load(meta_path.parent, self.extractor)
            self.models[meta["name"]] = (model, meta)
        if not self.models:
            raise SystemExit("No trained models in artifacts/. Run: python -m src.train")

    def inspect(self, name: str, image: Image.Image, truth: str | None = None) -> dict:
        model, meta = self.models[name]
        started = time.perf_counter()
        scores, maps = model.predict([image])
        ms = (time.perf_counter() - started) * 1000
        score = float(scores[0]) / meta["threshold"]
        heat = maps[0]
        y, x = np.unravel_index(int(np.argmax(heat)), heat.shape)
        result = {
            "product": name, "score": score, "defect": score > 1, "ms": ms,
            "hotspot": {"x": (x + 0.5) / heat.shape[1], "y": (y + 0.5) / heat.shape[0]},
            "image": data_url(view(image, model.crop, model.align).resize((SHOW, SHOW)), "JPEG"),
            "heat": data_url(heat_layer(heat, meta["threshold"]).resize((SHOW, SHOW), Image.BILINEAR), "PNG"),
        }
        # OpenCV boxes around the hot areas, as fractions of the picture
        size = heat.shape[0]
        result["regions"] = [{"x": r["x"] / size, "y": r["y"] / size, "w": r["w"] / size, "h": r["h"] / size,
                              "area": r["area"], "peak": r["peak"]}
                             for r in defect_regions(heat, meta["threshold"])] if score > 1 else []
        if truth is not None:
            result |= {"truth": truth, "correct": (score > 1) == (truth != "good")}
        return result


def create_app() -> FastAPI:
    inspector = Inspector()
    app = FastAPI(title="Visual Inspection Demo", docs_url="/api/docs", redoc_url=None)

    def model_or_404(name: str):
        if name not in inspector.models:
            raise HTTPException(404, f"Unknown product '{name}'")
        return inspector.models[name]

    def sample_or_404(name: str, index: int) -> dict:
        _, meta = model_or_404(name)
        if not 0 <= index < len(meta["samples"]):
            raise HTTPException(404, "No such sample")
        return meta["samples"][index]

    @app.get("/api/models")
    def models():
        """Every product model with its test results, figures and sample images."""
        out = []
        for name, (model, meta) in inspector.models.items():
            out.append({
                "name": name, "source": meta["source"], "public": meta["source"].startswith("MVTec"),
                "metrics": meta["metrics"], "learned_from": meta["learned_from"],
                "calibration_images": meta["calibration_images"], "test_images": meta["test_images"],
                "memory_bank": meta["memory_bank"], "patches_seen": meta["patches_seen"], "crop": model.crop,
                "figures": [{"url": f"/reports/{name}/{file}", "caption": caption} for file, caption in meta["figures"]],
                "samples": [{"index": i, "kind": s["kind"], "url": f"/api/sample/{name}/{i}"}
                            for i, s in enumerate(meta["samples"])],
            })
        summary = "/reports/summary.png" if (REPORTS_DIR / "summary.png").exists() else None
        return {"models": out, "summary": summary}

    @app.get("/api/sample/{name}/{index}")
    def sample(name: str, index: int):
        return FileResponse(resolve(sample_or_404(name, index)["path"]))

    @app.post("/api/inspect/{name}/sample/{index}")
    def inspect_sample(name: str, index: int):
        """Inspect one of the test images; the answer includes the true label."""
        s = sample_or_404(name, index)
        return inspector.inspect(name, load_rgb(resolve(s["path"])), s["kind"])

    @app.post("/api/inspect/{name}")
    def inspect_upload(name: str, file: UploadFile = File(...)):
        """Inspect an uploaded picture."""
        model_or_404(name)
        try:
            image = Image.open(io.BytesIO(file.file.read()))
            image.load()
        except (UnidentifiedImageError, OSError):
            raise HTTPException(400, "That file is not an image the server can read. Try PNG, JPG or BMP.")
        return inspector.inspect(name, load_rgb(image))

    app.mount("/reports", StaticFiles(directory=REPORTS_DIR), name="reports")
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


def free_port(start: int) -> int:
    for port in range(start, start + 20):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise SystemExit(f"No free port between {start} and {start + 19}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    print("Loading the pretrained network and the trained models ...")
    application = create_app()
    port = free_port(args.port)
    url = f"http://127.0.0.1:{port}"
    print(f"Demo running at {url}  (close this window or press Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, [url]).start()
    uvicorn.run(application, host="127.0.0.1", port=port, log_level="warning")
