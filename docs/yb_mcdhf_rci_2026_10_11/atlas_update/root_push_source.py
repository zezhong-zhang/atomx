"""Push the reviewed separate website commit using an ephemeral stdin credential."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

SITE = Path("/home/zzhang/.local/share/atomx-nist-atlas/site")
PROJECT = "appgprj_6aca6d7b3a588191bcfa3af5b091801f"


def git(*args):
    return subprocess.run(["git", *args], cwd=SITE, text=True,
                          capture_output=True, check=True).stdout.strip()


def main():
    payload = json.loads(sys.stdin.readline())
    credential = payload["credential"]
    assert json.loads((SITE / ".openai/hosting.json").read_text())["project_id"] == PROJECT
    assert git("rev-parse", "HEAD") == payload["commit_sha"]
    assert not git("status", "--porcelain"), "Website changed after reviewed commit"
    assert git("remote", "get-url", "origin") == credential["remote_url"]
    token = credential["token"]
    environment = os.environ.copy()
    environment["ATOMX_SITES_PUSH_TOKEN"] = token
    environment["GIT_TERMINAL_PROMPT"] = "0"
    environment["GIT_CONFIG_COUNT"] = "2"
    environment["GIT_CONFIG_KEY_0"] = "credential.helper"
    environment["GIT_CONFIG_VALUE_0"] = ""
    environment["GIT_CONFIG_KEY_1"] = "http.extraHeader"
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
    result = subprocess.run(["git", *auth_args, "push", "origin",
                             "HEAD:refs/heads/" + credential["branch"]],
                            cwd=SITE, env=environment, text=True, capture_output=True,
                            check=False)
    # Credentials remain only in process memory; diagnostics are redacted.
    if result.returncode:
        diagnostic = (result.stdout + result.stderr).replace(token, "[redacted]")
        print(json.dumps({"pushed": False, "returncode": result.returncode,
                          "diagnostic": diagnostic}))
        sys.exit(result.returncode)
    print(json.dumps({"pushed": True, "commit_sha": payload["commit_sha"],
                      "branch": credential["branch"], "credentials_persisted": False}))


if __name__ == "__main__":
    main()
