"""Push only the reviewed website commit using a credential held in memory."""

import hashlib
import json
import os
import shlex
import subprocess
import sys
import tarfile
import termios
from pathlib import Path

HERE = Path(__file__).resolve().parent
SITE = Path("/home/zzhang/.local/share/atomx-nist-atlas/site")
PROJECT = "appgprj_6aca6d7b3a588191bcfa3af5b091801f"


def git(*args, env=None):
    return subprocess.run(["git", *args], cwd=SITE, env=env, text=True,
                          capture_output=True, check=True).stdout.strip()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    # Suppress terminal echo before receiving the ephemeral stdin JSON.
    settings = termios.tcgetattr(sys.stdin.fileno())
    settings[3] &= ~termios.ECHO
    termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, settings)
    print("Ready for ephemeral credential; terminal echo disabled.", flush=True)
    payload = json.loads(sys.stdin.readline())
    credential, commit = payload["credential"], payload["commit_sha"]
    handoff = json.loads((HERE / "root_publication_preflight.json").read_text())
    hosting = json.loads((SITE / ".openai/hosting.json").read_text())
    assert hosting["project_id"] == PROJECT
    assert git("rev-parse", "HEAD") == commit
    assert not git("status", "--porcelain"), "Website changed after reviewed commit"
    assert git("remote", "get-url", "origin") == credential["remote_url"]
    assert credential["branch"] == "main"
    for path, expected in handoff["site_changed_and_added_files_sha256"].items():
        assert sha(SITE / path) == expected, path
    token = credential["token"]
    environment = os.environ.copy()
    environment.update(ATOMX_SITES_PUSH_TOKEN=token, GIT_TERMINAL_PROMPT="0",
                       GIT_CONFIG_COUNT="2", GIT_CONFIG_KEY_0="credential.helper",
                       GIT_CONFIG_VALUE_0="", GIT_CONFIG_KEY_1="http.extraHeader")
    if credential["auth_mode"] in {"bearer", "http_extra_header"}:
        environment["GIT_CONFIG_VALUE_1"] = "Authorization: Bearer " + token
        auth_args = []
    else:
        import base64
        environment["GIT_CONFIG_VALUE_1"] = "Authorization: Basic " + base64.b64encode(
            ("x-access-token:" + token).encode()).decode()
        helper = "!python3 -c " + shlex.quote(
            'import os; print("username=x-access-token"); '
            'print("password="+os.environ["ATOMX_SITES_PUSH_TOKEN"]); print()')
        auth_args = ["-c", "credential.helper=" + helper]
    command = ["git", *auth_args, "push", "origin", "HEAD:refs/heads/main"]
    pushed = subprocess.run(command, cwd=SITE, env=environment, text=True,
                            capture_output=True, check=False)
    if pushed.returncode:
        print(json.dumps({"pushed": False, "returncode": pushed.returncode,
                          "diagnostic": (pushed.stdout + pushed.stderr).replace(token, "[redacted]")}))
        sys.exit(pushed.returncode)
    remote = git(*auth_args, "ls-remote", "origin", "refs/heads/main", env=environment)
    assert remote.split()[0] == commit, "Remote source state does not match"
    archive = HERE / "site.tar"
    assert not archive.exists()
    subprocess.run(["git", "archive", "--format=tar", "--output=" + str(archive),
                    commit, "dist", ".openai/hosting.json"], cwd=SITE, check=True)
    with tarfile.open(archive) as packed:
        for path, expected in handoff["site_changed_and_added_files_sha256"].items():
            assert hashlib.sha256(packed.extractfile(path).read()).hexdigest() == expected
        assert hashlib.sha256(packed.extractfile(".openai/hosting.json").read()).hexdigest() == handoff["hosting_manifest_sha256"]
    receipt = {"pushed": True, "commit_sha": commit, "remote_HEAD_verified": True,
               "credentials_persisted": False, "branch": "main",
               "archive_sha256": sha(archive), "archive_bytes": archive.stat().st_size,
               "archive_built_from_pushed_commit": True, "Atomx_source_committed": False}
    with (HERE / "root_push_and_archive.json").open("x") as stream:
        stream.write(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
