from pathlib import Path

from typer.testing import CliRunner

from coinfosim.cli import app
from coinfosim.publish.publisher import PublishError, PublishResult


runner = CliRunner()


def test_publish_options_are_forwarded_and_auto_push_is_removed(tmp_path, monkeypatch):
    captured = {}

    def fake_publish(output_dir, **kwargs):
        captured["output_dir"] = output_dir
        captured.update(kwargs)
        return PublishResult(
            branch=kwargs["branch"],
            remote=kwargs["remote"],
            report_count=1,
            changed=True,
            pushed=False,
            commit_sha=None,
            reports=("nested/occupancy_scenario_report.html",),
        )

    monkeypatch.setattr("coinfosim.publish.publisher.publish_to_pages", fake_publish)
    output = tmp_path / "experiment-output"
    result = runner.invoke(
        app,
        [
            "publish",
            "--branch",
            "pages-test",
            "--remote",
            "upstream-test",
            "--output-dir",
            str(output),
            "--prune",
            "--dry-run",
            "--title",
            "Research Results",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "output_dir": output.resolve(),
        "branch": "pages-test",
        "remote": "upstream-test",
        "site_title": "Research Results",
        "include_data": True,
        "prune": True,
        "dry_run": True,
    }
    assert "Scenario reports: 1" in result.output
    assert "no Git changes were made" in result.output

    removed = runner.invoke(app, ["publish", "--auto-push"])
    assert removed.exit_code != 0
    assert "No such option" in removed.output


def test_publish_error_exits_with_status_one(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise PublishError("broken report link")

    monkeypatch.setattr("coinfosim.publish.publisher.publish_to_pages", fail)
    result = runner.invoke(app, ["publish", "--output-dir", str(tmp_path)])

    assert result.exit_code == 1
    assert "Publishing failed: broken report link" in result.output
