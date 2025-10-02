import os, tempfile
from flask import Blueprint, request, jsonify
from core.preprocess import read_stack_plus_mndwi
from core.postprocess import logits_to_mask_np, mask_to_png_b64
from core.config import THRESH
from models.unet13 import UNet13Inference
import io
import numpy as np
import tifffile as tiff
from PIL import Image
from flask import Blueprint, request, jsonify, url_for, send_file, make_response


def _ensure_hwc(arr):
    if arr.ndim == 2:
        return arr[..., None]
    if arr.ndim == 3 and arr.shape[0] <= 12 and arr.shape[0] < arr.shape[-1]:
        return np.transpose(arr, (1, 2, 0))
    return arr

def _rgb_composite_uint8(tif_path, rgb_idx=(3,2,1)):
    arr = tiff.imread(tif_path).astype(np.float32)
    hwc = _ensure_hwc(arr)
    rgb = hwc[..., list(rgb_idx)]
    p2, p98 = np.percentile(rgb, (2, 98))
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)
    return (rgb * 255).astype(np.uint8)  # (H,W,3) uint8

api = Blueprint("api", __name__)

_model = UNet13Inference()

@api.route("/health", methods=["GET"])
def health():
    return {"status": "ok", "threshold": THRESH}

@api.route("/predict", methods=["POST"])
def predict():
    """
    Accepts: multipart/form-data with file field "file" (TIFF)
             optional form field "threshold" (float)
    Returns: JSON {height, width, threshold, png_base64}
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded; expected form field 'file'"}), 400

    thr = float(request.form.get("threshold", THRESH))

    f = request.files["file"]
   
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    try:
        chw = read_stack_plus_mndwi(tmp_path)         # (13,H,W) np.float32
        logits = _model.predict_logits_from_np(chw)   # (1,1,H,W) torch tensor
        mask01 = logits_to_mask_np(logits, thr=thr)   # (H,W) float32 {0,1}
        h, w = mask01.shape
        b64 = mask_to_png_b64(mask01)
        return jsonify({
            "height": int(h), "width": int(w),
            "threshold": thr,
            "png_base64": b64
        })
    finally:
        try: os.remove(tmp_path)
        except Exception: pass

@api.get("/")
def index():
    return {
        "service": "WaterSeg API (PyTorch 13-channel)",
        "endpoints": {
            "GET /health": "status & default threshold",
            "POST /predict": "multipart/form-data with file=<TIFF>",
            "GET /form": "tiny upload form for manual testing"
        }
    }



@api.get("/form")
def upload_form():
    img_url = url_for('static', filename='satellite_photo.png')  
    return f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <title>Water Segmentation</title>
      <style>
        :root {{ --blue:#2563eb; }}
        body {{
          font-family: system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;
          margin:0; padding:0; background:#f8fafc;
        }}
        .container {{ max-width:1100px; margin:0 auto; padding:24px; }}
        h1 {{
          text-align:center; color:var(--blue);
          font-size:42px; font-weight:800; margin:16px 0 12px;
        }}
        .panel {{ width:70%; max-width:720px; margin:0 auto; }}
        @media (max-width: 780px) {{ .panel {{ width:95%; max-width:95%; }} }}
        .hero {{ margin-bottom:16px; }}
        .hero img {{
          width:100%; height:auto; border-radius:12px;
          box-shadow:0 8px 24px rgba(0,0,0,0.15);
        }}
        .panel .subtitle, .panel h2, .panel label {{ color:var(--blue); text-align:left; }}
        .subtitle {{ font-size:18px; margin:16px 0 8px; }}
        h2 {{ font-size:20px; margin:22px 0 8px; }}
        form {{ display:flex; align-items:center; gap:10px; margin:10px 0 12px; flex-wrap:wrap; }}
        input[type="number"] {{ width:80px; }}
        button {{
          padding:8px 12px; border-radius:8px; border:1px solid var(--blue);
          background:#fff; color:var(--blue); cursor:pointer;
        }}
        button:hover {{ background:#eff6ff; }}
        .result {{ margin:8px 0 18px; display:none; }}
        .result img {{
          max-width:100%; height:auto; border-radius:8px;
          box-shadow:0 6px 16px rgba(0,0,0,0.12);
        }}
        .error {{ color:#b91c1c; font-size:14px; }}
      </style>
    </head>
    <body>
      <div class="container">
        <h1>Water Segmentation</h1>

        <div class="panel">
          <div class="hero">
            <img src="{img_url}" alt="Satellite Imagery for Water Segmentation">
          </div>

          <div class="subtitle">Upload your TIFF and view the prediction</div>

          <h2>Mask </h2>
          <form id="maskForm" method="post" enctype="multipart/form-data" action="/predict-image">
            <input type="file" name="file" accept=".tif,.tiff" required />
            <label>Threshold:</label>
            <input type="number" step="0.01" min="0" max="1" name="threshold" value="0.5" />
            <button type="submit" id="maskBtn">Get Mask</button>
          </form>
          <div id="maskErr" class="error"></div>
          <div id="maskResult" class="result">
            <img id="maskOut" alt="Mask result">
          </div>

          <h2>Overlay on RGB </h2>
          <form id="overlayForm" method="post" enctype="multipart/form-data" action="/predict-overlay">
            <input type="file" name="file" accept=".tif,.tiff" required />
            <label>Threshold:</label>
            <input type="number" step="0.01" min="0" max="1" name="threshold" value="0.5" />
            <button type="submit" id="overlayBtn">Get Overlay</button>
          </form>
          <div id="overlayErr" class="error"></div>
          <div id="overlayResult" class="result">
            <img id="overlayOut" alt="Overlay result">
          </div>
        </div>
      </div>

      <script>
        async function wireForm(formId, btnId, outImgId, resultDivId, errDivId) {{
          const form = document.getElementById(formId);
          const btn  = document.getElementById(btnId);
          const out  = document.getElementById(outImgId);
          const resd = document.getElementById(resultDivId);
          const err  = document.getElementById(errDivId);

          form.addEventListener('submit', async (e) => {{
            e.preventDefault();
            err.textContent = '';
            btn.disabled = true;
            const original = btn.textContent;
            btn.textContent = 'Predicting…';

            try {{
              const fd = new FormData(form);
              // cache-buster so each threshold change fetches a fresh image
              const resp = await fetch(form.action + '?t=' + Date.now(), {{
                method: 'POST',
                body: fd
              }});
              if (!resp.ok) {{
                const txt = await resp.text();
                throw new Error('Server error: ' + txt);
              }}
              const blob = await resp.blob();
              const url = URL.createObjectURL(blob);
              out.src = url;
              resd.style.display = 'block';
            }} catch (ex) {{
              err.textContent = ex.message || String(ex);
              resd.style.display = 'none';
            }} finally {{
              btn.disabled = false;
              btn.textContent = original;
            }}
          }});
        }}

        wireForm('maskForm', 'maskBtn', 'maskOut', 'maskResult', 'maskErr');
        wireForm('overlayForm', 'overlayBtn', 'overlayOut', 'overlayResult', 'overlayErr');
      </script>
    </body>
    </html>
    """


@api.post("/predict-image")
def predict_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded; expected form field 'file'"}), 400

    try:
        thr = float(request.form.get("threshold", THRESH))
    except Exception:
        thr = THRESH
    thr = max(0.0, min(1.0, thr))
    print(f"[predict-image] threshold={thr}")

    f = request.files["file"]

    import tempfile, os, io
    from PIL import Image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    try:
        chw = read_stack_plus_mndwi(tmp_path)
        logits = _model.predict_logits_from_np(chw)  
        mask01 = logits_to_mask_np(logits, thr=thr)   

        im = Image.fromarray((mask01 * 255).astype(np.uint8), mode="L")
        buf = io.BytesIO()
        im.save(buf, format="PNG"); buf.seek(0)

        resp = make_response(send_file(buf, mimetype="image/png", download_name="mask.png"))
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"]        = "no-cache"
        resp.headers["Expires"]       = "0"
        resp.headers["X-Threshold"]   = str(thr)
        return resp
    finally:
        try: os.remove(tmp_path)
        except Exception: pass


@api.post("/predict-overlay")
def predict_overlay():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded; expected form field 'file'"}), 400

    try:
        thr = float(request.form.get("threshold", THRESH))
    except Exception:
        thr = THRESH
    thr = max(0.0, min(1.0, thr))
    print(f"[predict-overlay] threshold={thr}")

    f = request.files["file"]

    import tempfile, os, io, numpy as np
    from PIL import Image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    try:
        chw = read_stack_plus_mndwi(tmp_path)
        logits = _model.predict_logits_from_np(chw)
        mask01 = logits_to_mask_np(logits, thr=thr)

        rgb = _rgb_composite_uint8(tmp_path)
        alpha = 0.35
        color = np.array([255, 0, 0], dtype=np.float32) / 255.0
        base  = rgb.astype(np.float32) / 255.0
        m3    = mask01[..., None]
        overlay = base * (1 - alpha * m3) + color * (alpha * m3)
        overlay = (overlay * 255).astype(np.uint8)

        im = Image.fromarray(overlay, mode="RGB")
        buf = io.BytesIO()
        im.save(buf, format="PNG"); buf.seek(0)

        resp = make_response(send_file(buf, mimetype="image/png", download_name="overlay.png"))
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"]        = "no-cache"
        resp.headers["Expires"]       = "0"
        resp.headers["X-Threshold"]   = str(thr)
        return resp
    finally:
        try: os.remove(tmp_path)
        except Exception: pass
