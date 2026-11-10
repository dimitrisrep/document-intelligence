import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response

from document_intelligence.contracts import MAX_BYTES
from document_intelligence.service import DocumentExtractor, csv_export


def create_app(service=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.service = service or DocumentExtractor(
            Path(os.environ.get("DOCUMENT_ARTIFACTS", "artifacts"))
        )
        yield

    app = FastAPI(title="Document Intelligence", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok", "model_ready": app.state.service.model is not None}

    @app.get("/ready")
    def ready():
        if app.state.service.model is None:
            raise HTTPException(503, "No trained extractor loaded.")
        return {"status": "ready"}

    @app.post("/extract")
    async def extract(
        request: Request, format: str = Query("json", pattern="^(json|csv)$")
    ):
        # A raw byte stream keeps the hard limit effective before buffering an upload.
        content = bytearray()
        async for block in request.stream():
            if len(content) + len(block) > MAX_BYTES:
                raise HTTPException(413, "Document exceeds 8 MiB.")
            content.extend(block)
        try:
            # OCR is blocking: run it off the asynchronous request loop.
            from starlette.concurrency import run_in_threadpool

            result = await run_in_threadpool(app.state.service.extract, bytes(content))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (RuntimeError, OSError) as exc:
            raise HTTPException(
                503,
                "Extractor or OCR unavailable; check local artifacts and Tesseract.",
            ) from exc
        if format == "csv":
            return Response(
                csv_export(result),
                media_type="text/csv",
                headers={
                    "Content-Disposition": 'attachment; filename="extracted_fields.csv"'
                },
            )
        return result

    @app.get("/", response_class=HTMLResponse)
    def demo():
        return """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Document Intelligence</title><style>body{font-family:system-ui,sans-serif;max-width:860px;margin:48px auto;padding:0 20px;color:#202124}button,input{font:inherit;padding:10px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f4f4;padding:20px}</style>
<h1>Document Intelligence</h1><p>Extract receipt fields with local OCR and a learned text/layout classifier. Review every result against its evidence.</p>
<form id="form"><label>Receipt (PDF or image, up to 8 MiB) <input id="file" type="file" accept=".pdf,.png,.jpg,.jpeg,.tif,.tiff,.webp" required></label> <button>Extract</button></form><pre id="result" aria-live="polite">Ready.</pre>
<script>document.getElementById('form').onsubmit=async e=>{e.preventDefault();const out=document.getElementById('result');const file=document.getElementById('file').files[0];if(file.size>8*1024*1024){out.textContent='File exceeds 8 MiB.';return}out.textContent='Reading document...';try{const r=await fetch('/extract',{method:'POST',headers:{'Content-Type':'application/octet-stream'},body:file});out.textContent=JSON.stringify(await r.json(),null,2)}catch(e){out.textContent=String(e)}};</script></html>"""

    return app


app = create_app()
