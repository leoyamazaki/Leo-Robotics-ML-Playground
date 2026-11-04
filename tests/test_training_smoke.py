import json

from simple_arm.training_smoke import run_smoke


def test_cpu_training_and_artifacts(tmp_path):
    metrics = run_smoke(tmp_path)
    assert metrics["device"] == "cpu"
    assert metrics["validation_mse"][-1] < 0.25 * metrics["validation_mse"][0]
    assert metrics["checkpoint_verified"]
    assert (tmp_path / "model.pt").stat().st_size > 0
    assert (tmp_path / "loss.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    saved = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert saved == metrics
