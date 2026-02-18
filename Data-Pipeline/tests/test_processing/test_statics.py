from pathlib import Path
from scripts.processing.process_static import load_and_process_static

def test_static_cache(tmp_path: Path):
    out1 = load_and_process_static(64, str(tmp_path))
    assert out1.exists()

    # second call should reuse cache
    out2 = load_and_process_static(64, str(tmp_path))
    assert out2 == out1
