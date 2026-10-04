"""Tests for --darask-plugin-mode (see docs/SPEC.md §55 in the darask-paint repo).

These tests exercise the API layer only: the real ModelManager (and therefore
any actual torch inference / model download) is replaced with a lightweight
stub, following the same "mock the heavy model, test the routing/response
contract" approach used elsewhere in this test suite.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from iopaint import __version__
from iopaint.const import DARASK_PLUGIN_MODE_MODEL
from iopaint.helper import encode_pil_to_base64
from iopaint.schema import ApiConfig, Device, InteractiveSegModel, ModelInfo, ModelType, RealESRGANModel


class _FakeModelManager:
    """Stand-in for iopaint.model_manager.ModelManager.

    Avoids loading a real torch model (no download, no GPU, no inference)
    while still satisfying everything api.py touches on the object.
    """

    def __init__(self, name, device=None, **kwargs):
        self.name = name
        self.device = device
        self.enable_controlnet = False
        self.controlnet_method = None

    def __call__(self, image, mask, config):
        # Real models return a BGR np.ndarray the same size as the input.
        return image[:, :, ::-1].copy()

    def scan_models(self):
        return []

    def switch(self, new_name):
        self.name = new_name

    @property
    def current_model(self) -> ModelInfo:
        return ModelInfo(name=self.name, path=self.name, model_type=ModelType.INPAINT)


def _base_config(**overrides) -> ApiConfig:
    data = dict(
        host="127.0.0.1",
        port=8423,
        inbrowser=False,
        model="lama",
        no_half=False,
        low_mem=False,
        cpu_offload=False,
        disable_nsfw_checker=False,
        local_files_only=False,
        cpu_textencoder=False,
        device=Device.cpu,
        input=None,
        mask_dir=None,
        output_dir=None,
        quality=100,
        enable_interactive_seg=False,
        interactive_seg_model=InteractiveSegModel.vit_b,
        interactive_seg_device=Device.cpu,
        enable_remove_bg=False,
        remove_bg_device=Device.cpu,
        remove_bg_model="briaai/RMBG-1.4",
        enable_anime_seg=False,
        enable_realesrgan=False,
        realesrgan_device=Device.cpu,
        realesrgan_model=RealESRGANModel.realesr_general_x4v3,
        enable_gfpgan=False,
        gfpgan_device=Device.cpu,
        enable_restoreformer=False,
        restoreformer_device=Device.cpu,
        darask_plugin_mode=False,
    )
    data.update(overrides)
    return ApiConfig(**data)


def _make_app(config: ApiConfig) -> FastAPI:
    # Mirror cli.py's FastAPI() construction exactly: plugin mode disables
    # docs_url/redoc_url/openapi_url so /docs, /redoc, /openapi.json and
    # /docs/oauth2-redirect are never registered in the first place.
    if config.darask_plugin_mode:
        return FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    return FastAPI()


def _make_client(config: ApiConfig):
    from iopaint.api import Api

    app = _make_app(config)
    # Normal (non-plugin) mode mounts StaticFiles(WEB_APP_DIR); the built
    # frontend bundle isn't part of a source checkout, so point it at a
    # directory that is guaranteed to exist. Plugin mode never touches this.
    with patch("iopaint.api.ModelManager", _FakeModelManager), patch(
        "iopaint.api.WEB_APP_DIR", Path(__file__).parent
    ):
        api = Api(app, config)
    # base_url="http://127.0.0.1" so the default Host header on every request
    # satisfies plugin mode's DNS-rebinding guard; individual tests override
    # the Host/Origin headers explicitly to exercise that guard itself.
    return TestClient(app, base_url="http://127.0.0.1"), api


def _sample_image_b64() -> str:
    img = Image.new("RGB", (8, 8), color=(255, 0, 0))
    return encode_pil_to_base64(img, quality=100, infos={}).decode()


def _sample_mask_b64() -> str:
    mask = Image.new("L", (8, 8), color=255)
    return encode_pil_to_base64(mask, quality=100, infos={}).decode()


# ---------------------------------------------------------------------------
# 1. GET /api/v1/health matches the contract darask-paint reads.
# ---------------------------------------------------------------------------


def test_darask_plugin_mode_health():
    config = _base_config(darask_plugin_mode=True, model="lama")
    client, _api = _make_client(config)

    resp = client.get("/api/v1/health")

    assert resp.status_code == 200
    assert resp.json() == {
        "plugin": "darask-iopaint",
        "api": 1,
        "engine": __version__,
        "backend": "ready",
        "model": "lama",
    }


def test_darask_plugin_mode_health_reports_configured_model():
    config = _base_config(darask_plugin_mode=True, model="lama")
    client, _api = _make_client(config)

    resp = client.get("/api/v1/health")

    assert resp.json()["model"] == "lama"


# ---------------------------------------------------------------------------
# 2. POST /api/v1/inpaint exists and works with the stubbed model.
# ---------------------------------------------------------------------------


def test_darask_plugin_mode_inpaint_route_exists():
    config = _base_config(darask_plugin_mode=True, model="lama")
    client, _api = _make_client(config)

    resp = client.post(
        "/api/v1/inpaint",
        json={"image": _sample_image_b64(), "mask": _sample_mask_b64()},
    )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/")
    assert len(resp.content) > 0


def test_darask_plugin_mode_inpaint_rejects_extra_fields():
    # The plugin-mode request schema is image/mask only (extra="forbid");
    # the full IOpaint tuning surface (sd_steps, ldm_sampler, ...) isn't
    # meaningful for a hard-locked LaMa model and must be rejected.
    config = _base_config(darask_plugin_mode=True, model="lama")
    client, _api = _make_client(config)

    resp = client.post(
        "/api/v1/inpaint",
        json={"image": _sample_image_b64(), "mask": _sample_mask_b64(), "sd_steps": 50},
    )

    assert resp.status_code == 422


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"image": None, "mask": None},
        {"image": "", "mask": ""},
        {"mask": None},
        {"image": None},
    ],
)
def test_darask_plugin_mode_inpaint_missing_fields_are_400_not_422_or_500(body):
    config = _base_config(darask_plugin_mode=True, model="lama")
    client, _api = _make_client(config)

    resp = client.post("/api/v1/inpaint", json=body)

    assert resp.status_code == 400


def test_darask_plugin_mode_inpaint_bad_base64_is_400():
    config = _base_config(darask_plugin_mode=True, model="lama")
    client, _api = _make_client(config)

    resp = client.post(
        "/api/v1/inpaint",
        json={"image": "not-valid-base64!!!", "mask": _sample_mask_b64()},
    )

    assert resp.status_code == 400


def test_darask_plugin_mode_inpaint_undecodable_image_is_400():
    import base64

    config = _base_config(darask_plugin_mode=True, model="lama")
    client, _api = _make_client(config)
    garbage_but_valid_base64 = base64.b64encode(b"not an image, just bytes").decode()

    resp = client.post(
        "/api/v1/inpaint",
        json={"image": garbage_but_valid_base64, "mask": _sample_mask_b64()},
    )

    assert resp.status_code == 400


def test_darask_plugin_mode_inpaint_rejects_mismatched_model_defensively():
    config = _base_config(darask_plugin_mode=True, model="lama")
    client, api = _make_client(config)
    # Simulate the model somehow drifting away from the hard-locked value;
    # there is no route that can do this, but the handler must still guard.
    api.model_manager.name = "mat"

    resp = client.post(
        "/api/v1/inpaint",
        json={"image": _sample_image_b64(), "mask": _sample_mask_b64()},
    )

    # A server-side invariant broke (not a client input problem), so this is
    # 503 (was 409 in an earlier revision -- see gpt-5.6-sol review).
    assert resp.status_code == 503


def test_darask_plugin_mode_keeps_structured_error_json():
    # CORS is disabled, but the JSON exception-handling middleware (which
    # gives callers a predictable {"error", "detail", ...} body instead of
    # a bare Starlette error page) must still be active in plugin mode.
    config = _base_config(darask_plugin_mode=True, model="lama")
    client, api = _make_client(config)
    api.model_manager.name = "mat"  # force the 503 defensive check

    resp = client.post(
        "/api/v1/inpaint",
        json={"image": _sample_image_b64(), "mask": _sample_mask_b64()},
    )

    assert resp.status_code == 503
    body = resp.json()
    assert body["error"] == "HTTPException"
    assert DARASK_PLUGIN_MODE_MODEL in body["detail"]


def test_darask_plugin_mode_inpaint_size_mismatch_is_400():
    config = _base_config(darask_plugin_mode=True, model="lama")
    client, _api = _make_client(config)
    small_mask = Image.new("L", (4, 4), color=255)

    resp = client.post(
        "/api/v1/inpaint",
        json={
            "image": _sample_image_b64(),
            "mask": encode_pil_to_base64(small_mask, quality=100, infos={}).decode(),
        },
    )

    assert resp.status_code == 400


def test_darask_plugin_mode_inpaint_is_serialized():
    # inpaint calls must run one at a time (darask_inpaint_lock), even when
    # two requests arrive concurrently.
    config = _base_config(darask_plugin_mode=True, model="lama")
    client, _api = _make_client(config)

    state = {"current": 0, "max": 0}
    state_lock = threading.Lock()

    def slow_call(self, image, mask, cfg):
        with state_lock:
            state["current"] += 1
            state["max"] = max(state["max"], state["current"])
        time.sleep(0.3)
        with state_lock:
            state["current"] -= 1
        return image[:, :, ::-1].copy()

    payload = {"image": _sample_image_b64(), "mask": _sample_mask_b64()}

    with patch.object(_FakeModelManager, "__call__", slow_call):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(client.post, "/api/v1/inpaint", json=payload)
                for _ in range(2)
            ]
            results = [f.result() for f in futures]

    assert all(r.status_code == 200 for r in results)
    assert state["max"] == 1


# ---------------------------------------------------------------------------
# 3. Everything else is unregistered -> 404, and the route set is exact.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/"),
        ("GET", "/index.html"),
        ("GET", "/ws"),
        ("GET", "/ws/"),
        ("GET", "/api/v1/model"),
        ("POST", "/api/v1/model"),
        ("POST", "/api/v1/switch_plugin_model"),
        ("POST", "/api/v1/run_plugin_gen_mask"),
        ("POST", "/api/v1/run_plugin_gen_image"),
        ("POST", "/api/v1/segment_by_text"),
        ("GET", "/api/v1/server-config"),
        ("GET", "/api/v1/samplers"),
        ("GET", "/api/v1/inputimage"),
        ("POST", "/api/v1/save_image"),
        ("POST", "/api/v1/adjust_mask"),
        ("POST", "/api/v1/gen-info"),
        ("GET", "/api/v1/media_file"),
        ("GET", "/api/v1/medias"),
        ("GET", "/docs"),
        ("GET", "/redoc"),
        ("GET", "/openapi.json"),
        ("GET", "/docs/oauth2-redirect"),
    ],
)
def test_darask_plugin_mode_disabled_routes_are_404(method, path):
    config = _base_config(darask_plugin_mode=True)
    client, _api = _make_client(config)

    resp = client.request(method, path)

    assert resp.status_code == 404, f"{method} {path} should be 404, got {resp.status_code}"


@pytest.mark.parametrize("path", ["/api/v1/health", "/api/v1/inpaint"])
@pytest.mark.parametrize("method", ["HEAD", "OPTIONS"])
def test_darask_plugin_mode_wrong_method_is_405(method, path):
    config = _base_config(darask_plugin_mode=True)
    client, _api = _make_client(config)

    resp = client.request(method, path)

    assert resp.status_code == 405


def test_darask_plugin_mode_app_routes_are_exactly_health_and_inpaint():
    config = _base_config(darask_plugin_mode=True)
    _client, api = _make_client(config)

    routes = {
        (route.path, tuple(sorted(getattr(route, "methods", None) or [])))
        for route in api.app.routes
    }

    assert routes == {
        ("/api/v1/health", ("GET",)),
        ("/api/v1/inpaint", ("POST",)),
    }


def test_darask_plugin_mode_strips_preexisting_docs_routes_defensively():
    # Even if something hands Api() a FastAPI app that *did* register docs
    # (i.e. didn't go through cli.py's docs_url=None construction), plugin
    # mode must still end up with only the two API routes.
    config = _base_config(darask_plugin_mode=True)
    from iopaint.api import Api

    app = FastAPI()  # docs enabled by default this time
    assert any(r.path == "/docs" for r in app.routes)

    with patch("iopaint.api.ModelManager", _FakeModelManager), patch(
        "iopaint.api.WEB_APP_DIR", Path(__file__).parent
    ):
        api = Api(app, config)

    paths = {r.path for r in api.app.routes}
    assert paths == {"/api/v1/health", "/api/v1/inpaint"}


# ---------------------------------------------------------------------------
# 4. No CORS middleware; DNS-rebinding guard on Host/Origin.
# ---------------------------------------------------------------------------


def test_darask_plugin_mode_no_cors_headers_without_origin():
    config = _base_config(darask_plugin_mode=True)
    client, _api = _make_client(config)

    resp = client.get("/api/v1/health")

    assert resp.status_code == 200
    assert "access-control-allow-origin" not in {k.lower() for k in resp.headers.keys()}


@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost:12345",
        "http://127.0.0.1:5173",
        "http://[::1]:3000",
        "tauri://localhost",
        "null",
    ],
)
def test_darask_plugin_mode_rejects_local_browser_origin(origin):
    config = _base_config(darask_plugin_mode=True)
    client, _api = _make_client(config)

    resp = client.post("/api/v1/inpaint", headers={"Origin": origin}, content=b"{}")

    assert resp.status_code == 403
    assert resp.json()["error"] == "Forbidden"


def test_darask_plugin_mode_rejects_cross_origin_request():
    config = _base_config(darask_plugin_mode=True)
    client, _api = _make_client(config)

    resp = client.get("/api/v1/health", headers={"Origin": "http://example.com"})

    assert resp.status_code == 403
    assert resp.json()["error"] == "Forbidden"


def test_darask_plugin_mode_rejects_cross_origin_preflight():
    config = _base_config(darask_plugin_mode=True)
    client, _api = _make_client(config)

    resp = client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://example.com",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert resp.status_code == 403
    assert "access-control-allow-origin" not in {k.lower() for k in resp.headers.keys()}


@pytest.mark.parametrize(
    "bad_host",
    [
        "evil.com",
        "attacker.example",
        "127.0.0.1.evil.com",
        "0.0.0.0",
        "",
    ],
)
def test_darask_plugin_mode_rejects_dns_rebinding_host_header(bad_host):
    config = _base_config(darask_plugin_mode=True)
    client, _api = _make_client(config)

    resp = client.get("/api/v1/health", headers={"Host": bad_host})

    assert resp.status_code == 400
    assert resp.json()["error"] == "InvalidHost"


@pytest.mark.parametrize(
    "good_host",
    ["127.0.0.1", "127.0.0.1:8423", "localhost", "localhost:8423"],
)
def test_darask_plugin_mode_accepts_loopback_host_header(good_host):
    config = _base_config(darask_plugin_mode=True)
    client, _api = _make_client(config)

    resp = client.get("/api/v1/health", headers={"Host": good_host})

    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 5. Startup-time hard locks (host + model) raised directly by Api().
# ---------------------------------------------------------------------------


def test_darask_plugin_mode_refuses_non_loopback_host_at_construction():
    from iopaint.api import Api

    config = _base_config(darask_plugin_mode=True, host="0.0.0.0")
    app = _make_app(config)

    with patch("iopaint.api.ModelManager", _FakeModelManager):
        with pytest.raises(RuntimeError):
            Api(app, config)


def test_darask_plugin_mode_refuses_non_lama_model_at_construction():
    from iopaint.api import Api

    config = _base_config(darask_plugin_mode=True, model="mat")
    app = _make_app(config)

    with patch("iopaint.api.ModelManager", _FakeModelManager):
        with pytest.raises(RuntimeError):
            Api(app, config)


# ---------------------------------------------------------------------------
# 6. Normal (non-plugin) mode is unaffected.
# ---------------------------------------------------------------------------


def test_normal_mode_legacy_routes_still_registered():
    config = _base_config(darask_plugin_mode=False, model="lama")
    client, _api = _make_client(config)

    resp = client.get("/api/v1/model")

    assert resp.status_code == 200
    assert resp.json()["name"] == "lama"


def test_normal_mode_has_no_health_route():
    # /api/v1/health is a plugin-mode-only addition.
    config = _base_config(darask_plugin_mode=False, model="lama")
    client, _api = _make_client(config)

    resp = client.get("/api/v1/health")

    assert resp.status_code == 404


def test_normal_mode_has_cors_headers():
    config = _base_config(darask_plugin_mode=False, model="lama")
    client, _api = _make_client(config)

    resp = client.get(
        "/api/v1/model", headers={"Origin": "http://example.com"}
    )

    # allow_origins=["*"] + allow_credentials=True makes CORSMiddleware echo
    # the request Origin back (per the CORS spec) rather than send a literal
    # "*"; either way, the header's presence is what plugin mode must not have.
    assert resp.headers.get("access-control-allow-origin") == "http://example.com"


def test_normal_mode_inpaint_still_works():
    config = _base_config(darask_plugin_mode=False, model="lama")
    client, _api = _make_client(config)

    resp = client.post(
        "/api/v1/inpaint",
        json={"image": _sample_image_b64(), "mask": _sample_mask_b64()},
    )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/")


def test_normal_mode_model_switch_route_works():
    config = _base_config(darask_plugin_mode=False, model="lama")
    client, api = _make_client(config)

    resp = client.post("/api/v1/model", json={"name": "lama"})

    assert resp.status_code == 200
    assert resp.json()["name"] == "lama"
    assert api.model_manager.name == "lama"


def test_normal_mode_run_plugin_routes_are_reachable():
    # No plugins are enabled in this config, so the handler itself returns a
    # structured 422 ("Plugin not found") -- the point here is just that the
    # route exists and is wired up (unlike plugin mode, where it's a 404).
    config = _base_config(darask_plugin_mode=False, model="lama")
    client, _api = _make_client(config)

    resp = client.post(
        "/api/v1/run_plugin_gen_mask",
        json={"name": "RemoveBG", "image": _sample_image_b64()},
    )

    assert resp.status_code == 422
    assert resp.json()["detail"] == "Plugin not found"


def test_normal_mode_static_mount_serves_files():
    # StaticFiles(WEB_APP_DIR) is mounted at "/"; _make_client points
    # WEB_APP_DIR at this tests/ directory, so this file itself should be
    # served back, proving the mount is live (not just "not 404").
    config = _base_config(darask_plugin_mode=False, model="lama")
    client, _api = _make_client(config)

    resp = client.get("/test_darask_plugin_mode.py")

    assert resp.status_code == 200
    assert b"test_normal_mode_static_mount_serves_files" in resp.content
