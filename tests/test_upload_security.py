import io
from unittest.mock import patch
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from backend.main import app, MAX_UPLOAD_BYTES

client = TestClient(app)

def sample(size=(60,60)):
    out = io.BytesIO()
    Image.new("RGB", size, "white").save(out, format="PNG")
    return out.getvalue()

@pytest.mark.parametrize("url", ["/process-image", "/optimize"])
def test_reject_nonimage(url):
    r=client.post(url, files={"file":("fake.png",b"not an image","image/png")})
    assert r.status_code == 415

@pytest.mark.parametrize("url", ["/process-image", "/optimize"])
def test_reject_large_file(url):
    r=client.post(url, files={"file":("huge.png",b"x"*(MAX_UPLOAD_BYTES+1),"image/png")})
    assert r.status_code == 413

def test_reject_extreme_aspect_ratio():
    r=client.post("/optimize", files={"file":("a.png",sample((1,100)),"image/png")})
    assert r.status_code == 413

@pytest.mark.parametrize("key,value",[("handle_x","nan"),("handle_y","inf"),("handle_x","-1"),("target_mass","0"),("ball_mass","-1")])
def test_reject_bad_numbers(key,value):
    r=client.post("/optimize",data={key:value}, files={"file":("ok.png",sample(),"image/png")})
    assert r.status_code == 422

def test_optimization_success():
    with patch("backend.main.mechanics.run_optimization",return_value={"optimized":{"max_spin":1.0}}):
        r=client.post("/optimize", files={"file":("ok.png",sample(),"image/png")})
    assert r.status_code == 200
    assert r.json()["optimized"]["max_spin"] == 1.0

def test_internal_error_not_exposed():
    with patch("backend.main.mechanics.run_optimization",side_effect=RuntimeError("hidden-secret")):
        r=client.post("/optimize", files={"file":("ok.png",sample(),"image/png")})
    assert r.status_code == 500
    assert "hidden-secret" not in r.text
