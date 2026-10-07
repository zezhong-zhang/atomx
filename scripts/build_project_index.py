#!/usr/bin/env python3
"""Build a compact agent map and a queryable AST index, without importing AtomX."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from urllib.parse import quote

OUTPUTS = ("PROJECT_INDEX.md", "PROJECT_INDEX.json")
MAP_MAX_BYTES = 8 * 1024


def git(root: Path, *args: str, data: bytes | None = None) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *args], input=data, check=True, capture_output=True
    ).stdout


def kind(path: str) -> str | None:
    parts = PurePosixPath(path).parts
    if path in OUTPUTS or any(
        p in {"archive", "output", "__pycache__"} or p.startswith("source_")
        for p in parts
    ):
        return None
    if path.endswith(".py"):
        if path.startswith("src/atomx/"):
            return "source"
        if path.startswith("tests/") and (
            parts[-1].startswith("test_") or parts[-1] == "conftest.py"
        ):
            return "test"
        if path.startswith(("scripts/", "benchmarks/", "examples/", "docs/gallery/")):
            return "tool"
    if path.endswith(".md") and (
        path.startswith("docs/") or path in {"README.md", "AGENTS.md"}
    ):
        return "doc"
    if path in {"pyproject.toml", ".pre-commit-config.yaml"} or path.startswith(
        ".github/workflows/"
    ):
        return "config"
    return None


def read_inputs(root: Path, staged: bool) -> dict[str, bytes]:
    if not staged:
        names = git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
        result = {}
        for raw in names.split(b"\0"):
            path = raw.decode("utf-8")
            source = root / path
            if kind(path) and source.is_file() and not source.is_symlink():
                result[path] = source.read_bytes()
        return result
    entries = []
    for raw in git(root, "ls-files", "--stage", "-z").split(b"\0"):
        if not raw:
            continue
        metadata, path_raw = raw.split(b"\t", 1)
        mode, oid, stage = metadata.split()
        if stage != b"0":
            raise ValueError("Resolve Git index conflicts before building the project index")
        path = path_raw.decode("utf-8")
        if kind(path) and mode != b"120000":
            entries.append((path, oid))
    # One Git process reads all staged blobs, including partially staged files.
    output = git(root, "cat-file", "--batch", data=b"".join(oid + b"\n" for _, oid in entries))
    result = {}
    offset = 0
    for path, _ in entries:
        header_end = output.index(b"\n", offset)
        _, object_type, size = output[offset:header_end].split()
        if object_type != b"blob":
            raise ValueError(f"Expected a staged blob: {path}")
        start = header_end + 1
        offset = start + int(size)
        result[path] = output[start:offset]
        offset += 1
    return result


def module_name(path: str) -> str:
    if not path.startswith("src/"):
        return ""
    name = path[4:-3].replace("/", ".")
    return name.removesuffix(".__init__")


def describe(path: str, content: bytes) -> dict:
    category = kind(path)
    item = {"path": path, "kind": category, "sha256": hashlib.sha256(content).hexdigest()}
    if path.endswith(".py"):
        tree = ast.parse(content, filename=path)
        module = module_name(path)
        item["module"] = module
        item["summary"] = (ast.get_docstring(tree) or "").split("\n", 1)[0][:160]
        symbols = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                symbols.append({"name": node.name, "line": node.lineno})
                if isinstance(node, ast.ClassDef):
                    symbols.extend(
                        {"name": f"{node.name}.{method.name}", "line": method.lineno}
                        for method in node.body
                        if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
                    )
        item["symbols"] = symbols
        imports = set()
        package = module if path.endswith("/__init__.py") else module.rpartition(".")[0]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(
                    alias.name for alias in node.names
                    if alias.name == "atomx" or alias.name.startswith("atomx.")
                )
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level and package:
                    parent = package.split(".")[: len(package.split(".")) - node.level + 1]
                    base = ".".join([*parent, *([base] if base else [])])
                if base == "atomx" or base.startswith("atomx."):
                    imports.update(f"{base}.{alias.name}" for alias in node.names)
        item["imports"] = sorted(imports)
    elif category == "doc":
        lines = content.decode("utf-8").splitlines()
        item["summary"] = next((line[2:].strip() for line in lines if line.startswith("# ")), "")[:160]
    return item


def build(inputs: dict[str, bytes]) -> dict[str, bytes]:
    records = [describe(path, content) for path, content in sorted(inputs.items())]
    counts = Counter(record["kind"] for record in records)
    groups = Counter()
    for record in records:
        if record["kind"] == "source":
            parts = PurePosixPath(record["path"]).parts
            groups["/".join(parts[:3]) if len(parts) > 3 else "src/atomx"] += 1
    lines = [
        "# Project index: atomx",
        "",
        "自动生成的定位地图；先读此页，再查询相关符号/模块，最后读取命中的源码和测试。",
        "不要把完整 JSON 或全部文档加载到上下文；索引不证明物理能力或测试覆盖。",
        "",
        "## 按需查询",
        "",
        "```sh",
        "python scripts/build_project_index.py --query CILevel",
        "python scripts/build_project_index.py --query numerics/bspline --limit 10",
        "python scripts/build_project_index.py --tests src/atomx/atom.py --limit 10",
        "```",
        "",
        "查询只读 PROJECT_INDEX.json，返回路径/行号和有限条匹配；再用 rg 缩小读取范围。",
        "行号属于建索引时的快照；动态导入、重导出及间接测试需另查，静态候选不是覆盖证明。",
        "",
        "## 入口",
        "",
    ]
    for path in (
        "AGENTS.md", "src/atomx/__init__.py", "src/atomx/atom.py", "src/atomx/atom_inputs.py",
        "src/atomx/calculation_spec.py", "docs/README.md", "docs/ARCHITECTURE_DESIGN.md",
        "docs/ARCHITECTURE_CAPABILITIES.md", "docs/TESTING.md", "docs/AGENT_REFERENCE.md",
    ):
        if path in inputs:
            lines.append(f"- [{path}]({quote(path)})")
    lines += ["", "## 源码分区", "", "| 查询路径 | Python 文件数 |", "|---|---:|"]
    for path, count in sorted(groups.items()):
        lines.append(f"| [{path}/]({quote(path)}/) | {count} |")
    lines += [
        "", "## 其他上下文", "",
        f"- 测试 {counts['test']}，工具/示例 {counts['tool']}，文档 {counts['doc']}，配置 {counts['config']} 个文件。",
        "- 详细路径、AST 定义/类方法、静态内部导入和内容 SHA 存在 PROJECT_INDEX.json（schema 2）。",
        "- 不含归档目录、source_* 源码快照和运行输出；保留的专题文档仍须按正文识别适用范围。",
        "- 名称匹配无结果时直接 rg；只查询源代码结构，不推断实现状态、正确性或性能。",
        "", "## 更新", "",
        "```sh",
        "python scripts/build_project_index.py                 # 当前工作树，含未忽略的新文件",
        "python scripts/build_project_index.py --check         # 检查工作树索引是否过期",
        "python scripts/build_project_index.py --staged --stage # 提交 hook：只读暂存内容，只暂存两份索引",
        "```",
        "",
        "提交前由 pre-commit 自动更新；新克隆先运行 .venv/bin/pre-commit install。",
        "暂存模式不读取未暂存源码/文档，不导入 atomx、不运行 SCF；输出无时间戳，相同内容生成相同索引。",
        "索引有未暂存改动时 hook 拒绝覆盖：先核对并暂存/保存这些改动，再重新提交。",
        "文档修改/合并/删除仍按 AGENTS.md 执行，重建项目索引不代替文档语义维护。",
    ]
    # One compact record per line: searchable by tooling, without a huge pretty-printed tree.
    payload = '{"schema_version":2,"files":[\n' + ",\n".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in records
    ) + "\n]}\n"
    markdown = ("\n".join(lines) + "\n").encode()
    if len(markdown) > MAP_MAX_BYTES:
        raise ValueError("Project map exceeds 8 KiB; reduce the overview before expanding it")
    return {OUTPUTS[0]: markdown, OUTPUTS[1]: payload.encode()}


def publish(root: Path, outputs: dict[str, bytes], stage: bool) -> None:
    if stage:
        # Validate both destinations before writing either. Never overwrite local edits.
        for path in OUTPUTS:
            tracked = bool(git(root, "ls-files", "--", path))
            destination = root / path
            if destination.is_symlink():
                raise ValueError(f"Refusing generated-output symlink: {path}")
            if tracked:
                staged = git(root, "show", f":{path}")
                if not destination.exists() or destination.read_bytes() != staged:
                    raise ValueError(f"Unstaged index changes in {path}; review and stage/save them first")
            elif destination.exists():
                raise ValueError(f"Untracked output {path}; review and stage/save it first")
    for path, content in outputs.items():
        destination = root / path
        if destination.is_symlink():
            raise ValueError(f"Refusing generated-output symlink: {path}")
        if not destination.exists() or destination.read_bytes() != content:
            destination.write_bytes(content)
    if stage:
        git(root, "add", "--", *OUTPUTS)


def query(root: Path, term: str | None, tests: str | None, limit: int) -> None:
    payload = json.loads((root / OUTPUTS[1]).read_text())
    if payload.get("schema_version") != 2:
        raise ValueError("Rebuild the legacy project index before querying")
    matches = []
    for record in payload["files"]:
        path = record["path"]
        if tests:
            module = module_name(tests) if tests.startswith("src/") else tests
            if record["kind"] == "test" and any(
                imp == module or imp.startswith(module + ".")
                for imp in record.get("imports", [])
            ):
                matches.append((0, path, 1, "static import candidate"))
        else:
            needle = (term or "").casefold()
            if needle in path.casefold():
                matches.append((2, path, 1, record.get("summary", "")))
            for symbol in record.get("symbols", []):
                name = symbol["name"]
                if needle in name.casefold():
                    rank = 0 if name.casefold().split(".")[-1] == needle else 1
                    matches.append((rank, path, symbol["line"], name))
    for _, path, line, label in sorted(matches)[:limit]:
        print(f"{path}:{line}  {label}")
    print(f"{len(matches)} matches; showing at most {limit}. Verify against current files.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--stage", action="store_true")
    action.add_argument("--check", action="store_true")
    action.add_argument("--query")
    action.add_argument("--tests")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    if args.stage and not args.staged:
        parser.error("--stage requires --staged")
    if args.limit < 1 or (args.staged and (args.query or args.tests)):
        parser.error("Use a positive limit; queries read the saved index, not --staged")
    root = Path(git(Path.cwd(), "rev-parse", "--show-toplevel").decode().strip())
    try:
        if args.query is not None or args.tests is not None:
            query(root, args.query, args.tests, args.limit)
            return 0
        outputs = build(read_inputs(root, args.staged))
        if args.check:
            for path, expected in outputs.items():
                actual = git(root, "show", f":{path}") if args.staged else (root / path).read_bytes()
                if actual != expected:
                    print(f"Stale project index: {path}", file=sys.stderr)
                    return 1
        else:
            publish(root, outputs, args.stage)
        print(f"Project index {'verified' if args.check else 'rebuilt'} ({'staged' if args.staged else 'worktree'}).")
        return 0
    except (OSError, ValueError, SyntaxError, subprocess.CalledProcessError) as error:
        print(f"Project index: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
