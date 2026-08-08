import os
from pathlib import Path

import pytest

from voice_chat.config import Config, ROOT


def test_config_loads_from_env(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "\n".join(
            [
                "LLM_API_KEY=test-key",
                "LLM_BASE_URL=https://example.com/v1/",
                "LLM_MODEL=demo-model",
                "MOONSHINE_MODEL_ARCH=tiny_streaming",
                "COSYVOICE_FP16=true",
            ]
        )
    )
    cfg = Config.load(env)
    assert cfg.llm_api_key == "test-key"
    assert cfg.llm_base_url == "https://example.com/v1"
    assert cfg.llm_model == "demo-model"
    assert cfg.moonshine_model_arch == "tiny_streaming"
    assert cfg.cosyvoice_fp16 is True
    assert cfg.cosyvoice_model_dir.is_absolute()
    assert ROOT in cfg.cosyvoice_model_dir.parents or cfg.cosyvoice_model_dir == ROOT / "pretrained_models" / "CosyVoice2-0.5B"


def test_missing_api_key_exits(tmp_path):
    env = tmp_path / ".env"
    env.write_text("LLM_MODEL=x\n")
    # Clear any inherited key
    os.environ.pop("LLM_API_KEY", None)
    with pytest.raises(SystemExit):
        Config.load(env)
