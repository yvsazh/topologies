#!/usr/bin/env python3
"""Reject assistant attribution in identities, commit metadata, and repository files."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

_PARTS = (("clau", "de"), ("anthro", "pic"))
_SYNTH_PROVIDER_PARTS = (
    ("open", "ai"),
    ("ol", "lama"),
    ("v", "llm"),
    ("llama", "cpp"),
    ("lm", "studio"),
    ("open", "router"),
)


def _pattern(parts: tuple[str, ...], *, token_boundaries: bool = False) -> re.Pattern[str]:
    letters = "".join(parts)
    pattern = r"[ \t._-]*".join(re.escape(letter) for letter in letters)
    if token_boundaries:
        pattern = rf"(?<![A-Za-z0-9_]){pattern}(?![A-Za-z0-9_])"
    return re.compile(pattern, re.IGNORECASE)


_PATTERNS = tuple(_pattern(parts) for parts in _PARTS)
_ORGANIZATION_FILE_PATTERN = _pattern(_PARTS[1], token_boundaries=True)
_SYNTH_PROVIDER_PATTERNS = tuple(
    _pattern(parts, token_boundaries=True)
    for parts in (
        *_SYNTH_PROVIDER_PARTS,
        ("openai", "api", "key"),
        (*_PARTS[1], "api", "key"),
        ("openai", "compat"),
    )
)
_PERSON_NAME = re.compile(
    r"\s+(?:(?:(?i:de|del|da|di|dos|du|la|le|van|von|der|den|ten|ter|st)\.?\s+)|"
    r"(?:[A-Z]\.\s+))*[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?\b"
)
_CONTEXT = re.compile(
    r"\b(?:code|opus|sonnet|haiku|assistant|model|bot|generated|authored|co.?authored|"
    r"contributor|committer|written|helped|built|powered|settings?|hooks?|configuration|"
    r"prompt|signature|email|noreply|api|md)\b",
    re.IGNORECASE,
)
_EXPLICIT_CONTEXT = re.compile(
    r"\b(?:code|opus|sonnet|haiku|assistant|bot|generated|authored|co.?authored|"
    r"contributor|committer|written|helped|built|powered|settings?|hooks?|configuration|"
    r"prompt|signature|email|noreply|api|md)\b",
    re.IGNORECASE,
)
_MODEL_ID_FIELD = re.compile(
    r"(?ix)(?<![a-z0-9_.-])model[\s_-]*id\b[ \t]*[\"']?[ \t]*[:=][ \t]*"
    r"(?:\"(?:\\.|[^\"\\])*\"|'(?:''|[^'])*'|(?:(?![ \t]+\#)[^\r\n])*)"
)


def _matches(text: str, index: int) -> list[re.Match[str]]:
    return list(_PATTERNS[index].finditer(text))


def has_prohibited_message(text: str) -> bool:
    """Commit and tag messages must not contain these identifiers in any form."""
    return any(pattern.search(text) for pattern in _PATTERNS)


def has_prohibited_identity(identity: str) -> bool:
    """Reject assistant identities while allowing unrelated people with the same name."""
    if _PATTERNS[1].search(identity):
        return True

    assistant_matches = _matches(identity, 0)
    if not assistant_matches:
        return False

    display_name, _, email_part = identity.partition("<")
    display_name = display_name.strip()
    display_words = display_name.split()
    email = email_part.split(">", 1)[0]
    email_domain = email.partition("@")[2]
    has_human_name = bool(
        display_words and _PERSON_NAME.match(display_name[len(display_words[0]) :])
    )
    if has_human_name:
        display_has_assistant_name = bool(_PATTERNS[0].search(display_name))
        if display_has_assistant_name and not _PATTERNS[0].search(email_domain):
            return bool(_EXPLICIT_CONTEXT.search(display_name))
        return True
    if _CONTEXT.search(display_name):
        return True

    compact_name = re.sub(r"[^a-z0-9]", "", display_name.casefold())
    compact_assistant = "".join(_PARTS[0])
    if compact_name == compact_assistant:
        return True
    return bool(_PATTERNS[0].search(email))


def has_prohibited_file_reference(text: str, path: str = "") -> bool:
    """Find vendor/assistant references without flagging unrelated human names in prose."""
    searchable_text = _MODEL_ID_FIELD.sub("", text)
    if _ORGANIZATION_FILE_PATTERN.search(path) or _ORGANIZATION_FILE_PATTERN.search(searchable_text):
        return True

    if _is_synthesis_artifact(path) and any(
        pattern.search(path) or pattern.search(searchable_text)
        for pattern in _SYNTH_PROVIDER_PATTERNS
    ):
        return True

    for component in Path(path).parts:
        compact_component = re.sub(r"[^a-z0-9]", "", component.casefold())
        if compact_component == "".join(_PARTS[0]) or (
            _PATTERNS[0].search(component) and _CONTEXT.search(component)
        ):
            return True

    for line in searchable_text.splitlines():
        for match in _matches(line, 0):
            before = line[max(0, match.start() - 48) : match.start()]
            after = line[match.end() : match.end() + 72]
            if _CONTEXT.search(before + " " + after):
                return True
            if not _PERSON_NAME.match(after):
                return True
    return False


def _is_synthesis_artifact(path: str) -> bool:
    parts = PurePosixPath(path).parts
    if not parts:
        return False
    if parts[0] == "docs":
        return any(
            part == "synth"
            or part.startswith(("synth-", "synth_", "synth.", "synthesis"))
            for part in parts[1:]
        )
    if parts[0] == "tests":
        return any(
            part.startswith(("test_synth", "test_synthesis", "synth", "synthesis"))
            for part in parts[1:]
        )
    return False


def _git_bytes(*args: str, input_data: bytes | None = None) -> bytes:
    return subprocess.check_output(["git", *args], input=input_data)


def _tracked_paths(staged: bool) -> list[str]:
    if staged:
        result = _git_bytes("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z")
    else:
        result = _git_bytes("ls-files", "-z")
    return [os.fsdecode(path) for path in result.split(b"\0") if path]


def _scan_files(issues: list[str], staged: bool) -> None:
    for path in _tracked_paths(staged):
        try:
            data = _git_bytes("show", f":{path}") if staged else Path(path).read_bytes()
        except (OSError, subprocess.CalledProcessError):
            continue
        if b"\0" in data:
            continue
        text = data.decode("utf-8", errors="replace")
        if has_prohibited_file_reference(text, path):
            issues.append(f"file {path}: prohibited assistant/provider reference")


def _scan_identities(issues: list[str]) -> None:
    for field in ("GIT_AUTHOR_IDENT", "GIT_COMMITTER_IDENT"):
        try:
            identity = _git_bytes("var", field).decode("utf-8", errors="replace").strip()
        except subprocess.CalledProcessError:
            continue
        if has_prohibited_identity(identity):
            issues.append(f"configured {field.removeprefix('GIT_').lower()} is prohibited")


def _scan_commits(issues: list[str]) -> None:
    commits = _git_bytes("rev-list", "--all").splitlines()
    if not commits:
        return
    objects = _git_bytes("cat-file", "--batch", input_data=b"\n".join(commits) + b"\n")
    position = 0
    for oid in commits:
        header_end = objects.index(b"\n", position)
        header = objects[position:header_end].split()
        size = int(header[2])
        start = header_end + 1
        payload = objects[start : start + size]
        position = start + size + 1
        metadata, separator, message = payload.partition(b"\n\n")
        if not separator:
            metadata, message = payload, b""
        short_oid = oid.decode("ascii")[:12]
        for line in metadata.splitlines():
            if line.startswith((b"author ", b"committer ")):
                identity = line.split(b" ", 1)[1].decode("utf-8", errors="replace")
                if has_prohibited_identity(identity):
                    field = line.split(b" ", 1)[0].decode("ascii")
                    issues.append(f"commit {short_oid}: prohibited {field} identity")
        if has_prohibited_message(message.decode("utf-8", errors="replace")):
            issues.append(f"commit {short_oid}: prohibited assistant attribution in message")


def _cat_file_payloads(objects: list[tuple[bytes, int]]):
    batch: list[tuple[bytes, int]] = []
    batch_size = 0
    for object_id, size in objects:
        if batch and (len(batch) >= 128 or batch_size + size > 8 * 1024 * 1024):
            yield from _read_cat_file_batch(batch)
            batch = []
            batch_size = 0
        batch.append((object_id, size))
        batch_size += size
    if batch:
        yield from _read_cat_file_batch(batch)


def _read_cat_file_batch(objects: list[tuple[bytes, int]]):
    object_ids = [object_id for object_id, _ in objects]
    data = _git_bytes("cat-file", "--batch", input_data=b"\n".join(object_ids) + b"\n")
    position = 0
    for object_id, _ in objects:
        header_end = data.index(b"\n", position)
        size = int(data[position:header_end].split()[2])
        start = header_end + 1
        end = start + size
        yield object_id, data[start:end]
        position = end + 1


def _non_synthesis_history_paths(
    candidate_blob_ids: set[bytes], tree_objects: list[tuple[bytes, int]]
) -> dict[bytes, str]:
    if not candidate_blob_ids or not tree_objects:
        return {}

    hash_size = len(tree_objects[0][0]) // 2
    tree_entries: dict[bytes, list[tuple[bytes, bytes, bytes]]] = {}
    for tree_id, payload in _cat_file_payloads(tree_objects):
        entries: list[tuple[bytes, bytes, bytes]] = []
        entry_position = 0
        while entry_position < len(payload):
            mode_end = payload.index(b" ", entry_position)
            name_end = payload.index(b"\0", mode_end + 1)
            object_start = name_end + 1
            object_end = object_start + hash_size
            entries.append(
                (
                    payload[entry_position:mode_end],
                    payload[mode_end + 1 : name_end],
                    payload[object_start:object_end].hex().encode("ascii"),
                )
            )
            entry_position = object_end
        tree_entries[tree_id] = entries

    unresolved = set(candidate_blob_ids)
    roots = set(_git_bytes("log", "--all", "--reflog", "--format=%T").splitlines())
    refs = _git_bytes(
        "for-each-ref",
        "--format=%(objecttype)%09%(objectname)%09%(*objecttype)%09%(*objectname)",
    )
    for row in refs.splitlines():
        object_type, object_id, peeled_type, peeled_id = row.split(b"\t", 3)
        if object_type == b"tree":
            roots.add(object_id)
        if peeled_type == b"tree" and peeled_id:
            roots.add(peeled_id)

    non_synthesis_paths: dict[bytes, str] = {}
    tree_cache: dict[bytes, dict[bytes, tuple[str, ...]]] = {}

    def paths_in_tree(tree_id: bytes) -> dict[bytes, tuple[str, ...]]:
        if tree_id in tree_cache:
            return tree_cache[tree_id]
        paths: dict[bytes, list[str]] = {}
        for mode, name, object_id in tree_entries.get(tree_id, ()):
            decoded_name = os.fsdecode(name)
            if mode in (b"40000", b"040000"):
                for blob_id, nested_paths in paths_in_tree(object_id).items():
                    paths.setdefault(blob_id, []).extend(
                        f"{decoded_name}/{nested_path}" for nested_path in nested_paths
                    )
            elif object_id in candidate_blob_ids:
                paths.setdefault(object_id, []).append(decoded_name)
        result = {blob_id: tuple(found_paths) for blob_id, found_paths in paths.items()}
        if tree_id not in roots:
            tree_cache[tree_id] = result
        return result

    for root in roots:
        for object_id, found_paths in paths_in_tree(root).items():
            if object_id not in unresolved:
                continue
            for path in found_paths:
                if not _is_synthesis_artifact(path):
                    non_synthesis_paths[object_id] = path
                    unresolved.remove(object_id)
                    break
        if not unresolved:
            break
    return non_synthesis_paths


def _scan_history_blobs(issues: list[str]) -> None:
    objects = _git_bytes("rev-list", "--objects", "--all", "--reflog").splitlines()
    paths: dict[bytes, str] = {}
    object_ids: list[bytes] = []
    for row in objects:
        object_id, separator, path = row.partition(b" ")
        if object_id and object_id not in paths:
            paths[object_id] = os.fsdecode(path) if separator else ""
            object_ids.append(object_id)
    if not object_ids:
        return

    object_list = b"\n".join(object_ids) + b"\n"
    object_types = _git_bytes(
        "cat-file",
        "--batch-check=%(objectname) %(objecttype) %(objectsize)",
        input_data=object_list,
    )
    object_info = [row.split() for row in object_types.splitlines()]
    blob_objects = [(row[0], int(row[2])) for row in object_info if row[1] == b"blob"]
    tree_objects = [(row[0], int(row[2])) for row in object_info if row[1] == b"tree"]
    if not blob_objects:
        return
    blob_ids = [object_id for object_id, _ in blob_objects]
    synthesis_blob_ids = {
        object_id for object_id in blob_ids if _is_synthesis_artifact(paths.get(object_id, ""))
    }
    non_synthesis_paths = _non_synthesis_history_paths(synthesis_blob_ids, tree_objects)

    for object_id, payload in _cat_file_payloads(blob_objects):
        if b"\0" in payload:
            continue
        path = non_synthesis_paths.get(object_id, paths.get(object_id, ""))
        if _is_synthesis_artifact(path):
            # Older provider-negative tests and design prompts remain on explicitly
            # unrewritten legacy refs; current tracked files are checked separately.
            continue
        text = payload.decode("utf-8", errors="replace")
        if has_prohibited_file_reference(text, path):
            short_oid = object_id.decode("ascii")[:12]
            issues.append(f"historical blob {short_oid} ({path or 'unknown path'}): prohibited reference")


def _scan_refs(issues: list[str]) -> None:
    result = _git_bytes("for-each-ref", "--format=%(refname)%09%(objectname)%09%(objecttype)")
    for row in result.splitlines():
        fields = row.decode("utf-8", errors="replace").split("\t")
        if len(fields) != 3:
            continue
        refname, oid, object_type = fields
        if has_prohibited_message(refname):
            issues.append(f"ref {refname}: prohibited reference name")
        if object_type == "tag":
            tag = _git_bytes("cat-file", "tag", oid).decode("utf-8", errors="replace")
            if has_prohibited_message(tag):
                issues.append(f"tag {refname}: prohibited assistant attribution")
        elif object_type == "blob":
            data = _git_bytes("cat-file", "blob", oid)
            if b"\0" not in data and has_prohibited_file_reference(
                data.decode("utf-8", errors="replace"), refname
            ):
                issues.append(f"ref {refname}: prohibited file content")
        elif object_type == "tree":
            entries = _git_bytes("ls-tree", "-r", "-z", oid)
            for entry in entries.split(b"\0"):
                if not entry:
                    continue
                metadata, path = entry.split(b"\t", 1)
                fields = metadata.split()
                if len(fields) != 3 or fields[1] != b"blob":
                    continue
                data = _git_bytes("cat-file", "blob", fields[2].decode("ascii"))
                file_path = os.fsdecode(path)
                if b"\0" not in data and has_prohibited_file_reference(
                    data.decode("utf-8", errors="replace"), file_path
                ):
                    issues.append(f"tree ref {refname}: prohibited file {file_path}")


def _scan_message_file(issues: list[str], path: Path) -> None:
    message = path.read_text(encoding="utf-8", errors="replace")
    if has_prohibited_message(message):
        issues.append("commit message contains prohibited assistant attribution")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="check all refs, commits, tags, and tracked files")
    parser.add_argument("--staged", action="store_true", help="check the configured identities and staged files")
    parser.add_argument("--message-file", type=Path, help="check a commit message file")
    args = parser.parse_args()

    issues: list[str] = []
    if args.message_file:
        _scan_message_file(issues, args.message_file)
    elif args.staged:
        _scan_identities(issues)
        _scan_files(issues, staged=True)
    else:
        _scan_refs(issues)
        _scan_commits(issues)
        _scan_history_blobs(issues)
        _scan_files(issues, staged=False)

    if issues:
        for issue in issues:
            print(issue, file=sys.stderr)
        return 1
    print("Repository integrity check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
