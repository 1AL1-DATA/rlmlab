import json
import os
import shutil
import tempfile

import pytest

_tmp = tempfile.mkdtemp(prefix="rlmlab_test_")
os.environ["RLMLAB_HOME"] = os.path.join(_tmp, ".rlmlab")

from rlmlab import harness, subagents  # noqa: E402


@pytest.fixture(autouse=True)
def clean_state():
    """Isolate tests: fresh subagents dir + empty harness before each test."""
    shutil.rmtree(subagents.SUBAGENTS_DIR, ignore_errors=True)
    os.makedirs(harness.HARNESS_DIR, exist_ok=True)
    with open(harness.STATE_FILE, "w") as f:
        json.dump(harness.EMPTY_STATE, f, indent=2)
    yield
