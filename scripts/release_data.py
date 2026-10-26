"""Read only exact author-reviewed data files; never expands disclosure scope."""
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

MAX_STORED = 10 * 1024 * 1024
MAX_CONTENT = 64 * 1024 * 1024


def load_policy(root):
    path = root / 'data/RELEASE_DATA_POLICY.json'
    if not path.exists(): return {}
    if path.is_symlink(): raise ValueError('policy symlink')
    document = json.loads(path.read_text())
    if document['status'] != 'AUTHOR_AUTHORIZED_FIELD_REVIEWED_DISTRIBUTION_CANDIDATE':
        raise ValueError('invalid release policy status')
    entries = {}
    for row in document['files']:
        name = row['path']; relative = Path(name)
        if (relative.is_absolute() or '..' in relative.parts or name != relative.as_posix()
                or not name.startswith(('data/reproduction/', 'data/oof/')) or name in entries
                or row['kind'] not in {'patches', 'split', 'oof'}):
            raise ValueError('invalid reviewed data path/kind')
        entries[name] = row
    return entries


def read_reviewed(root, row):
    path = root / row['path']
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('unsafe release data path')
    if path.stat().st_size > MAX_STORED: raise ValueError('reviewed file too large')
    stored = path.read_bytes()
    if hashlib.sha256(stored).hexdigest() != row['sha256']: raise ValueError('reviewed stored SHA mismatch')
    if path.suffix == '.gz':
        with gzip.GzipFile(fileobj=io.BytesIO(stored)) as handle: content = handle.read(MAX_CONTENT + 1)
    else: content = stored
    if len(content) > MAX_CONTENT or hashlib.sha256(content).hexdigest() != row['content_sha256']:
        raise ValueError('reviewed content size/SHA mismatch')
    text = content.decode('utf-8')
    reader = csv.DictReader(io.StringIO(text, newline=''))
    if reader.fieldnames != row['columns'] or sum(1 for _ in reader) != row['rows']:
        raise ValueError('reviewed data schema/count mismatch')
    return content
