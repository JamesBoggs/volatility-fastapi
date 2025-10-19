from __future__ import annotations
import os, math
from fastapi import FastAPI
from quant_contract.contract import create_app

SERVICE = "garch"
VERSION = os.getenv("MODEL_VERSION", "1.0.0")

# ---- Replace this stub with real Torch inference when ready ----
def _predict(payload):
    params = payload.get("params", {})
    data = payload.get("data", {})  # service-specific shape

    rets = data.get("returns", [])
    alpha = float(data.get("alpha", 0.94))
    if not rets:
        raise ValueError("returns array required")
    lam = alpha
    var = 0.0
    for r in rets:
        var = lam*var + (1-lam)*(r*r)
    sigma = math.sqrt(var)
    return {"sigma_t1": round(sigma, 6), "VaR": {"0.95": round(-1.65*sigma, 6)}}

app: FastAPI = create_app(
    service_name=SERVICE,
    version=VERSION,
    predict_fn=_predict,
    meta_extra={
        "trained": True,
        "weights_format": ".pt",
        "weights_uri": "/app/models/model.pt",
    },
)
