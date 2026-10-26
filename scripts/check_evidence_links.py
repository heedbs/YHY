"""Validate Markdown file links and source-index digests without research data."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote


def safe_file(root, name):
    relative = Path(name)
    path = (root / relative).resolve()
    if relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(root):
        raise ValueError('unsafe repository reference')
    if not path.is_file(): raise ValueError('missing repository file')
    return path


def check(root):
    root = root.resolve(); findings = []; links = 0; records = 0
    for path in root.rglob('*.md'):
        if '.git' in path.relative_to(root).parts: continue
        for match in re.finditer(r'\[[^\]]*\]\(([^)]+)\)', path.read_text()):
            target = match.group(1).strip().strip('<>')
            if target.startswith(('https:', 'http:', 'mailto:', '#')): continue
            links += 1
            resolved = (path.parent / unquote(target.split('#')[0])).resolve()
            if not resolved.is_relative_to(root) or not resolved.exists():
                findings.append(dict(file=path.relative_to(root).as_posix(), rule='broken_or_escaping_markdown_file_link'))
    for name, path_key, hash_key in (
        ('docs/provenance/CLAIM_SOURCE_INDEX.csv', 'repository_path', 'repository_sha256'),
        ('docs/provenance/SOURCE_FILE_FINGERPRINTS.csv', 'public_path', 'public_sha256')):
        seen = set()
        try:
            with safe_file(root, name).open(newline='') as handle: rows = list(csv.DictReader(handle))
            if not rows: raise ValueError('empty index')
            for row in rows:
                target, expected = row[path_key], row[hash_key]
                if target in seen or not re.fullmatch(r'[0-9a-f]{64}', expected): raise ValueError('duplicate or invalid index')
                seen.add(target)
                if hashlib.sha256(safe_file(root, target).read_bytes()).hexdigest() != expected: raise ValueError('index SHA mismatch')
                records += 1
        except (ValueError, OSError, KeyError):
            findings.append(dict(file=name, rule='source_index_missing_invalid_or_mismatched'))
    return dict(markdown_file_links_checked=links, source_records_checked=records, findings=findings, passed=not findings)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('root', type=Path)
    args = parser.parse_args(); result = check(args.root)
    print(json.dumps(result, indent=2)); raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__': main()
