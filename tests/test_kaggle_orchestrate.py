"""Tests for kaggle_runner/orchestrate.py. Mocks subprocess.run so these
tests don't hit the real Kaggle API -- Task 6 does a real push/poll/pull
round trip against the live account as an execution-only step."""
import json
from unittest.mock import MagicMock, patch

import pytest

from kaggle_runner.orchestrate import (
    get_status,
    pull_output,
    push,
    render_kernel_script,
    wait_for_completion,
)


def test_render_kernel_script_bakes_in_commit_and_entrypoint():
    script = render_kernel_script("deadbeef", "experiments.calibrate_speed")
    assert 'REPO_COMMIT = "deadbeef"' in script
    assert 'ENTRYPOINT = "experiments.calibrate_speed"' in script


def test_push_renders_script_writes_it_and_calls_kaggle_kernels_push(tmp_path):
    metadata = {"id": "someuser/some-kernel"}
    (tmp_path / "kernel-metadata.json").write_text(json.dumps(metadata))
    with patch("kaggle_runner.orchestrate.subprocess.run") as mock_run:
        kernel_id = push(tmp_path, repo_commit="abc123", entrypoint="experiments.diagnose_label_bias")
    written_script = (tmp_path / "kernel_template.py").read_text()
    assert "abc123" in written_script
    assert "experiments.diagnose_label_bias" in written_script
    mock_run.assert_called_once_with(
        ["kaggle", "kernels", "push", "-p", str(tmp_path)], check=True
    )
    assert kernel_id == "someuser/some-kernel"


def test_get_status_parses_kaggle_cli_output():
    fake_result = MagicMock(stdout='someuser/some-kernel has status "complete"\n')
    with patch("kaggle_runner.orchestrate.subprocess.run", return_value=fake_result) as mock_run:
        status = get_status("someuser/some-kernel")
    mock_run.assert_called_once_with(
        ["kaggle", "kernels", "status", "someuser/some-kernel"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert status == "complete"


def test_wait_for_completion_polls_until_terminal_status():
    statuses = iter(["running", "running", "complete"])
    with patch("kaggle_runner.orchestrate.get_status", side_effect=lambda _id: next(statuses)):
        with patch("kaggle_runner.orchestrate.time.sleep") as mock_sleep:
            status = wait_for_completion("someuser/some-kernel", poll_interval_s=0.01, timeout_s=10)
    assert status == "complete"
    assert mock_sleep.call_count == 2


def test_wait_for_completion_raises_timeout_error_when_never_terminal():
    with patch("kaggle_runner.orchestrate.get_status", return_value="running"):
        with patch("kaggle_runner.orchestrate.time.monotonic", side_effect=[0.0, 0.0, 100.0]):
            with pytest.raises(TimeoutError):
                wait_for_completion("someuser/some-kernel", poll_interval_s=0.01, timeout_s=10)


def test_pull_output_calls_kaggle_kernels_output_and_creates_dest(tmp_path):
    dest = tmp_path / "out"
    with patch("kaggle_runner.orchestrate.subprocess.run") as mock_run:
        result = pull_output("someuser/some-kernel", dest)
    mock_run.assert_called_once_with(
        ["kaggle", "kernels", "output", "someuser/some-kernel", "-p", str(dest)],
        check=True,
    )
    assert result == dest
    assert dest.exists()
