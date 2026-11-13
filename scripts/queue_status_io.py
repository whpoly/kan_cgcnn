"""Atomic queue status writes tolerant of transient Windows reader locks."""
import json
import os
from pathlib import Path
import time
import uuid


def write_status(path, payload, attempts=30):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        for attempt in range(attempts):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt + 1 == attempts:
                    raise
                time.sleep(min(0.1 * (attempt + 1), 1.0))
    finally:
        if temporary.exists():
            temporary.unlink()
