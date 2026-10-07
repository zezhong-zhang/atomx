"""Project navigation must represent the committed snapshot without staging user edits."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_project_index.py"


def run(root, *args, check=True, env=None):
    return subprocess.run(
        args, cwd=root, check=check, text=True, capture_output=True, env=env
    )


def git(root, *args):
    return run(root, "git", *args).stdout


def build(root, *args, check=True):
    return run(root, sys.executable, str(SCRIPT), *args, check=check)


def records(root, staged=False):
    content = (
        git(root, "show", ":PROJECT_INDEX.json")
        if staged else (root / "PROJECT_INDEX.json").read_text()
    )
    return {item["path"]: item for item in json.loads(content)["files"]}


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Index Test")
    git(tmp_path, "config", "user.email", "index-test@example.invalid")
    git(tmp_path, "config", "commit.gpgsign", "false")
    source = tmp_path / "src/atomx"
    source.mkdir(parents=True)
    (source / "sample.py").write_text('"""Navigation fixture."""\ndef old_symbol():\n    pass\n')
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/old.md").write_text("# Old guide\n")
    git(tmp_path, "add", "src", "docs")
    build(tmp_path)
    git(tmp_path, "add", "PROJECT_INDEX.md", "PROJECT_INDEX.json")
    return tmp_path


def test_deterministic_ast_query_and_snapshot_exclusions(repo):
    source = repo / "src/atomx/sample.py"
    source.write_text("raise RuntimeError('must never import')\nclass Entry:\n    def solve(self): pass\n")
    old_snapshot = repo / "docs/experiment/source_final"
    old_snapshot.mkdir(parents=True)
    (old_snapshot / "AGENTS.md").write_text("# Duplicate snapshot\n")
    build(repo)
    before = (repo / "PROJECT_INDEX.json").read_bytes()
    build(repo)
    assert (repo / "PROJECT_INDEX.json").read_bytes() == before
    assert "docs/experiment/source_final/AGENTS.md" not in records(repo)
    result = build(repo, "--query", "solve", "--limit", "1")
    assert "src/atomx/sample.py:3  Entry.solve" in result.stdout
    assert len(result.stdout.splitlines()) == 2


def test_staged_rename_delete_partial_stage_and_untracked_files(repo):
    git(repo, "mv", "src/atomx/sample.py", "src/atomx/renamed.py")
    source = repo / "src/atomx/renamed.py"
    source.write_text("def staged_symbol(): pass\n")
    git(repo, "add", "src/atomx/renamed.py")
    source.write_text("def unstaged_symbol(): pass\n")
    (repo / "src/atomx/untracked.py").write_text("def not_committed(): pass\n")
    git(repo, "rm", "-f", "docs/old.md")
    before = git(repo, "ls-files", "--stage", "--", "src", "docs")
    build(repo, "--staged", "--stage")
    indexed = records(repo, staged=True)
    assert indexed["src/atomx/renamed.py"]["symbols"] == [{"name": "staged_symbol", "line": 1}]
    assert "src/atomx/sample.py" not in indexed
    assert "src/atomx/untracked.py" not in indexed
    assert "docs/old.md" not in indexed
    assert source.read_text() == "def unstaged_symbol(): pass\n"
    assert git(repo, "ls-files", "--stage", "--", "src", "docs") == before
    build(repo, "--staged", "--check")


def test_stage_refuses_local_index_edits_before_writing_any_output(repo):
    before_md = (repo / "PROJECT_INDEX.md").read_bytes()
    (repo / "PROJECT_INDEX.json").write_text("local edits\n")
    (repo / "src/atomx/sample.py").write_text("def updated(): pass\n")
    git(repo, "add", "src")
    result = build(repo, "--staged", "--stage", check=False)
    assert result.returncode == 1
    assert "Unstaged index changes" in result.stderr
    assert (repo / "PROJECT_INDEX.md").read_bytes() == before_md
    assert (repo / "PROJECT_INDEX.json").read_text() == "local edits\n"


def test_syntax_error_fails_without_publishing_partial_outputs(repo):
    before = (repo / "PROJECT_INDEX.json").read_bytes()
    (repo / "src/atomx/sample.py").write_text("def broken(\n")
    git(repo, "add", "src")
    result = build(repo, "--staged", "--stage", check=False)
    assert result.returncode == 1
    assert (repo / "PROJECT_INDEX.json").read_bytes() == before


def test_check_detects_source_change_and_static_test_candidates(repo):
    tests = repo / "tests"
    tests.mkdir()
    (tests / "test_sample.py").write_text("from atomx.sample import old_symbol\n")
    assert build(repo, "--check", check=False).returncode == 1
    build(repo)
    build(repo, "--check")
    output = build(repo, "--tests", "src/atomx/sample.py").stdout
    assert "tests/test_sample.py:1  static import candidate" in output


def test_actual_pre_commit_hook_rebuilds_and_stages_only_indexes(repo):
    config = repo / ".pre-commit-config.yaml"
    entry = f"{shlex.quote(sys.executable)} {shlex.quote(str(SCRIPT))} --staged --stage"
    config.write_text(
        "repos:\n  - repo: local\n    hooks:\n      - id: project-index\n"
        "        name: Project index\n"
        f"        entry: {json.dumps(entry)}\n"
        "        language: system\n        pass_filenames: false\n"
        "        always_run: true\n        stages: [pre-commit]\n"
    )
    git(repo, "add", ".pre-commit-config.yaml")
    build(repo, "--staged", "--stage")
    git(repo, "commit", "-qm", "initial fixture")
    env = dict(os.environ, PRE_COMMIT_HOME=str(repo / "hook-cache"))
    run(repo, sys.executable, "-m", "pre_commit", "install", env=env)
    source = repo / "src/atomx/sample.py"
    source.write_text("def committed_symbol(): pass\n")
    git(repo, "add", "src")
    source.write_text("def working_symbol(): pass\n")
    (repo / "src/atomx/untracked.py").write_text("def private_work(): pass\n")
    run(repo, "git", "commit", "-qm", "indexed fixture", env=env)
    committed = json.loads(git(repo, "show", "HEAD:PROJECT_INDEX.json"))
    indexed = {item["path"]: item for item in committed["files"]}
    assert indexed["src/atomx/sample.py"]["symbols"][0]["name"] == "committed_symbol"
    assert "src/atomx/untracked.py" not in indexed
    assert source.read_text() == "def working_symbol(): pass\n"
    assert git(repo, "diff", "--cached", "--name-only") == ""
