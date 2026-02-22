from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from app.api import router as api_router
from app.api.exception_handlers import install_error_handlers
from app.core.config import get_settings
from app.core.lifecycle import lifespan
from app.core.middlewares import install_middleware

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs" if settings.app_env != "prd" else None,
    redoc_url="/redoc" if settings.app_env != "prd" else None,
    openapi_url="/openapi.json" if settings.app_env != "prd" else None,
)

install_middleware(app)
install_error_handlers(app)
app.include_router(api_router, prefix=settings.api_prefix)


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    """Minimal single-page UI for demo inference."""
    return f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>{settings.app_name}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; max-width: 1100px; }}
    h1 {{ margin-bottom: 8px; }}
    .row {{ display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 16px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px; }}
    img {{ width: 100%; height: auto; border: 1px solid #eee; border-radius: 6px; }}
    .muted {{ color: #666; }}
    .result {{ margin-top: 16px; }}
    button {{ padding: 8px 12px; cursor: pointer; }}
    select, input[type=file] {{ padding: 6px; }}
    code {{ background: #f4f4f4; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>{settings.app_name}</h1>
  <p class="muted">Slug: <code>anomaly-localization-api</code></p>

  <div class="row">
    <input id="imageInput" type="file" accept="image/*" />
    <select id="methodSelect">
      <option value="padim">padim</option>
      <option value="student">student</option>
    </select>
    <button id="predictBtn">Predict</button>
  </div>

  <div id="status" class="muted"></div>

  <div id="result" class="result" style="display:none;">
    <div class="card" style="margin-bottom: 16px;">
      <div><strong>Method:</strong> <span id="outMethod"></span></div>
      <div><strong>Filename:</strong> <span id="outFilename"></span></div>
      <div><strong>Score:</strong> <span id="outScore"></span></div>
      <div><strong>Threshold:</strong> <span id="outThreshold"></span></div>
      <div><strong>Decision:</strong> <span id="outDecision"></span></div>
      <div class="muted" id="outNote"></div>
    </div>

    <div class="grid">
      <div class="card">
        <div><strong>Original</strong></div>
        <img id="imgOriginal" alt="Original image" />
      </div>
      <div class="card">
        <div><strong>Heatmap</strong></div>
        <img id="imgHeatmap" alt="Heatmap image" />
      </div>
      <div class="card">
        <div><strong>Overlay</strong></div>
        <img id="imgOverlay" alt="Overlay image" />
      </div>
    </div>
  </div>

  <script>
    const btn = document.getElementById("predictBtn");
    const imageInput = document.getElementById("imageInput");
    const methodSelect = document.getElementById("methodSelect");
    const statusEl = document.getElementById("status");
    const resultEl = document.getElementById("result");

    btn.addEventListener("click", async () => {{
      const file = imageInput.files[0];
      if (!file) {{
        statusEl.textContent = "Select an image first.";
        return;
      }}

      statusEl.textContent = "Running prediction...";
      resultEl.style.display = "none";

      const formData = new FormData();
      formData.append("image", file);
      formData.append("method", methodSelect.value);

      try {{
        const res = await fetch("/api/v1/predict", {{
          method: "POST",
          body: formData
        }});

        const data = await res.json();

        if (!res.ok) {{
          statusEl.textContent = data?.error?.message || "Request failed.";
          return;
        }}

        document.getElementById("outMethod").textContent = data.method;
        document.getElementById("outFilename").textContent = data.filename;
        document.getElementById("outScore").textContent = Number(data.image_score).toFixed(4);
        document.getElementById("outThreshold").textContent = Number(data.threshold).toFixed(2);
        document.getElementById("outDecision").textContent = data.is_anomalous ? "anomalous" : "normal";
        document.getElementById("outNote").textContent = data.note || "";

        const mime = data.content_type || "image/png";
        document.getElementById("imgOriginal").src = `data:${{mime}};base64,${{data.original_image_base64}}`;
        document.getElementById("imgHeatmap").src = `data:${{mime}};base64,${{data.heatmap_image_base64}}`;
        document.getElementById("imgOverlay").src = `data:${{mime}};base64,${{data.overlay_image_base64}}`;

        resultEl.style.display = "block";
        statusEl.textContent = "Done.";
      }} catch (err) {{
        statusEl.textContent = "Request error: " + (err?.message || "unknown");
      }}
    }});
  </script>
</body>
</html>
"""
