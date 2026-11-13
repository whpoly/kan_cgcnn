"""Resume the authors' public archive, verify HTTP ranges, and extract scoped files.

Only regular phonons files and small source/metadata files are extracted. No
archive code is executed, no links are created, and existing files are verified.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
import time
import urllib.request

URL = "https://ndownloader.figshare.com/files/57963316"
CHUNK = 64 * 1024 * 1024


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def request(start, end):
    req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}", "User-Agent": "research-reproduction"})
    result = urllib.request.urlopen(req, timeout=90)
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", result.headers.get("Content-Range", ""))
    if result.status != 206 or not match or (int(match[1]), int(match[2])) != (start, end):
        result.close()
        raise ValueError("Server did not honor the exact requested byte range.")
    return result, int(match[3])


def extract(archive, output):
    destination = (output / "extracted").resolve()
    destination.mkdir(exist_ok=True)
    inventory, saved = [], []
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            name = PurePosixPath(member.name)
            inventory.append(dict(name=member.name, size=member.size, type=str(member.type)))
            wanted = "phonon" in member.name.lower() or (name.suffix.lower() in (".py", ".sh", ".json", ".md", ".yml", ".yaml", ".txt") and member.size < 2 * 1024 * 1024)
            if not wanted or not member.isfile():
                continue
            if name.is_absolute() or ".." in name.parts or any(":" in p for p in name.parts):
                raise ValueError("Unsafe archive path: " + member.name)
            path = destination.joinpath(*name.parts).resolve()
            path.relative_to(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            sha = hashlib.sha256()
            temporary = path.with_suffix(path.suffix + ".extracting")
            with stream.extractfile(member) as source, temporary.open("wb") as target:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    target.write(chunk)
                    sha.update(chunk)
            if temporary.stat().st_size != member.size:
                raise ValueError("Truncated archive member.")
            if path.exists():
                old = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
                if old != sha.hexdigest():
                    raise ValueError("Existing extracted file differs: " + str(path))
                temporary.unlink()
            else:
                temporary.replace(path)
            saved.append(dict(name=member.name, path=str(path), size=member.size, sha256=sha.hexdigest()))
            print("EXTRACTED", member.name, member.size, flush=True)
            save(output / "extracted-files.json", saved)
    save(output / "archive-inventory.json", inventory)
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output.resolve()
    probe, total = request(0, 15)
    with probe:
        if not probe.read().startswith(b"\x1f\x8b"):
            raise ValueError("Not a gzip archive.")
    state_path = output / "download-status.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(url=URL, bytes=total, chunk_bytes=CHUNK, chunks={})
    if state["url"] != URL or state["bytes"] != total or state["chunk_bytes"] != CHUNK:
        raise ValueError("Archive metadata changed.")
    archive = output / "featurized_datasets.tar.gz"
    part = output / "featurized_datasets.tar.gz.part"
    state.update(state="downloading", pid=os.getpid())
    save(state_path, state)
    if not archive.exists():
        if not part.exists():
            with part.open("wb") as file:
                file.truncate(total)
        if part.stat().st_size != total:
            raise ValueError("Partial archive size changed.")
        with part.open("rb") as file:
            for key, expected in list(state["chunks"].items()):
                index = int(key)
                file.seek(index * CHUNK)
                actual = hashlib.sha256(file.read(min(CHUNK, total - index * CHUNK))).hexdigest()
                if actual != expected:
                    del state["chunks"][key]

        def fetch(index):
            start, end = index * CHUNK, min((index + 1) * CHUNK, total) - 1
            for attempt in range(4):
                try:
                    response, size = request(start, end)
                    if size != total:
                        raise ValueError("Archive size changed during download.")
                    count, sha = 0, hashlib.sha256()
                    with response, part.open("r+b") as file:
                        file.seek(start)
                        while True:
                            data = response.read(1024 * 1024)
                            if not data:
                                break
                            if count + len(data) > end - start + 1:
                                raise ValueError("Unexpected extra bytes.")
                            file.write(data)
                            count += len(data)
                            sha.update(data)
                    if count != end - start + 1:
                        raise ValueError("Truncated HTTP response.")
                    return index, sha.hexdigest()
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)

        count = (total + CHUNK - 1) // CHUNK
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fetch, i) for i in range(count) if str(i) not in state["chunks"]]
            for future in as_completed(futures):
                index, sha = future.result()
                state["chunks"][str(index)] = sha
                state["completed_chunks"], state["total_chunks"] = len(state["chunks"]), count
                state["updated_at"] = datetime.now(timezone.utc).isoformat()
                save(state_path, state)
                print("DOWNLOAD", len(state["chunks"]), "/", count, flush=True)
        part.replace(archive)
    with archive.open("rb") as file:
        state["archive_sha256"] = hashlib.file_digest(file, "sha256").hexdigest()
    state.update(state="extracting", verified_size=archive.stat().st_size)
    save(state_path, state)
    saved = extract(archive, output)
    state.update(state="complete", extracted_files=len(saved), completed_at=datetime.now(timezone.utc).isoformat())
    save(state_path, state)
    print("COMPLETE", output, flush=True)


if __name__ == "__main__":
    main()
