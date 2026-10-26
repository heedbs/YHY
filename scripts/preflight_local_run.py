"""Read-only input verification; optional CPU content audit. Never executes models."""
import argparse
import json
import os
from pathlib import Path

from configure_local_run import validate_stage
from prepare_local_manifests import read_rows, sha, validate_rows

ROOT = Path(__file__).resolve().parents[1]
LOCAL_FILES = {'final_unique_patch_manifest.csv', 'final_outer_inner_split_manifest.csv',
               'selected_model_pretrained_provenance.csv'}
SEMANTIC_FIELDS = ('uid', 'x', 'y', 'w', 'h', 'cls', 'patient_key', 'source_sha256')


def code_digest(stage):
    import hashlib
    root = ROOT / stage
    files = [root / 'train_one_run.py', *sorted((root / 'orchid').glob('*.py'))]
    if stage == 'stage1':
        files += [root / 'validate_before_run.py', root / 'evaluate_oof.py']
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(sha(path).encode())
    return digest.hexdigest()


def verify_local_rows(local, frozen):
    expected = {row['uid']: row for row in frozen}
    if len(expected) != len(frozen) or len(local) != len(frozen):
        raise ValueError('local patch count or frozen UID mismatch')
    seen = set()
    for row in local:
        uid = row['uid']
        if uid in seen or uid not in expected:
            raise ValueError('duplicate or unexpected local patch UID')
        seen.add(uid)
        ref = expected[uid]
        if any(row[k] != ref[k] for k in SEMANTIC_FIELDS):
            raise ValueError('local patch semantics differ from frozen bundle')
        relative = Path(ref['source_relative_path'])
        source, cache = Path(row['path']), Path(row['cache_path'])
        if (not source.is_absolute() or not cache.is_absolute()
                or '..' in relative.parts or relative.is_absolute()
                or tuple(source.parts[-len(relative.parts):]) != relative.parts
                or cache.name != uid + '.jpg'):
            raise ValueError('local source/cache binding mismatch')
        if not source.is_file() or not cache.is_file():
            raise ValueError('missing source or existing cache file')


def verify_content(rows, progress=None):
    """Hash original sources; decode existing cache only, never regenerate it."""
    import hashlib
    import cv2
    import numpy as np
    cv2.setNumThreads(1)
    sources = {}
    for row in rows:
        name = row['path']
        if sources.setdefault(name, row['source_sha256']) != row['source_sha256']:
            raise ValueError('contradictory source content identity')
    def stamp(path):
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
    for index, (name, expected) in enumerate(sorted(sources.items()), 1):
        path = Path(name); before = stamp(path)
        if sha(path) != expected or stamp(path) != before:
            raise ValueError('source SHA mismatch or concurrent source modification')
        if progress and index % 1000 == 0: progress('sources', index, len(sources))
    digest = hashlib.sha256()
    uids = set()
    for index, row in enumerate(sorted(rows, key=lambda r: r['uid']), 1):
        if row['uid'] in uids:
            raise ValueError('duplicate UID in content audit')
        uids.add(row['uid'])
        path = Path(row['cache_path']); before = stamp(path); payload = path.read_bytes()
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape != (int(row['h']), int(row['w']), 3):
            raise ValueError('cache decode or crop dimension mismatch')
        if stamp(path) != before:
            raise ValueError('concurrent cache modification')
        digest.update(row['uid'].encode() + b'\0' + hashlib.sha256(payload).hexdigest().encode() + b'\n')
        if progress and index % 10000 == 0: progress('cache', index, len(rows))
    return {'sources_sha_verified': len(sources), 'cache_decoded': len(rows),
            'cache_inventory_sha256': digest.hexdigest(),
            'cache_digest_definition': 'SHA256 of UID-sorted UTF8(uid) + NUL + ASCII(file_SHA256) + LF',
            'cache_generated': False, 'historical_first_generation_verified': False,
            'decoder': 'OpenCV IMREAD_COLOR; one CPU decoder thread',
            'opencv_version': cv2.__version__, 'numpy_version': np.__version__}


def check(stage, config_path, manifest_dir, bundle, output):
    if output.exists() or output.is_symlink():
        raise ValueError('output collision; never overwrite or auto-resume')
    identity = json.loads((ROOT / 'data/manifest_bundle_identity.json').read_text())
    for name, digest in identity['files'].items():
        if sha(bundle / name) != digest:
            raise ValueError('authorized bundle SHA mismatch')
    frozen = read_rows(bundle / 'patches.csv')
    splits = read_rows(bundle / 'final_outer_inner_split_manifest.csv')
    validate_rows(frozen, splits, identity['expected'])
    receipt = json.loads((manifest_dir / 'LOCAL_BINDING_RECEIPT.json').read_text())
    if set(receipt['files']) != LOCAL_FILES:
        raise ValueError('local receipt file coverage mismatch')
    for name in LOCAL_FILES:
        if sha(manifest_dir / name) != receipt['files'][name]:
            raise ValueError('local manifest receipt SHA mismatch')
    split_sha = sha(manifest_dir / 'final_outer_inner_split_manifest.csv')
    if split_sha != identity['files']['final_outer_inner_split_manifest.csv']:
        raise ValueError('exact frozen split SHA mismatch')
    local = read_rows(manifest_dir / 'final_unique_patch_manifest.csv')
    verify_local_rows(local, frozen)
    config = json.loads(config_path.read_text())
    validate_stage(stage, config)
    templates = [json.loads(p.read_text()) for p in (ROOT / 'configs' / stage).glob('*.json')]
    templates = [t for t in templates if t['run_id'] == config['run_id']]
    if len(templates) != 1 or {k: v for k, v in config.items() if k != 'code_sha256'} != templates[0]:
        raise ValueError('local config differs from frozen scientific template')
    if config.get('code_sha256') != code_digest(stage) or config['split_sha256'] != split_sha:
        raise ValueError('local code/config/split SHA mismatch')
    weights = read_rows(manifest_dir / 'selected_model_pretrained_provenance.csv')
    reference = read_rows(ROOT / 'data/pretrained_sources.csv')
    stripped = lambda rows: [{k: v for k, v in row.items() if k != 'local_cache_path'} for row in rows]
    if stripped(weights) != stripped(reference):
        raise ValueError('pretrained provenance differs from frozen reference')
    for row in weights:
        if sha(Path(row['local_cache_path'])) != row['weight_sha256']:
            raise ValueError('pretrained content SHA mismatch')
    key = 'EfficientNet-Lite2+Mamba' if stage == 'stage2' else config['model_name']
    selected = [row for row in weights if row['model_name'] == key]
    if len(selected) != 1 or selected[0]['weight_sha256'] != config['pretrained_sha256']:
        raise ValueError('config/pretrained identity mismatch')
    return {'status': 'PASS_LOCAL_INPUT_PREFLIGHT_ONLY', 'stage': stage,
            'config_sha256': sha(config_path), 'code_sha256': config['code_sha256'],
            'split_sha256': split_sha, 'pretrained_sha256': config['pretrained_sha256'],
            'patches': len(local), 'patients': identity['expected']['patients'],
            'output_collision': False, 'source_and_cache_presence_checked': True,
            'source_content_rehashed': False, 'cache_content_rehashed': False,
            'model_execution': False, 'gpu_access': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['stage1', 'stage2'], required=True)
    for name in ('config', 'manifest-dir', 'bundle', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--verify-content', action='store_true', help='Hash all source files and decode/hash all cache files on CPU; can take minutes')
    parser.add_argument('--expected-cache-digest', help='Require this whole-cache inventory SHA; requires --verify-content')
    args = parser.parse_args()
    env = os.environ.get('ORCHID_MANIFEST_DIR')
    if not env or Path(env).resolve() != args.manifest_dir.resolve():
        raise ValueError('ORCHID_MANIFEST_DIR must match --manifest-dir')
    if args.expected_cache_digest and not args.verify_content:
        parser.error('--expected-cache-digest requires --verify-content')
    result = check(args.stage, args.config, args.manifest_dir, args.bundle, args.output)
    if args.verify_content:
        result['content_audit'] = verify_content(read_rows(args.manifest_dir / 'final_unique_patch_manifest.csv'))
        result['source_content_rehashed'] = True
        result['cache_content_rehashed'] = True
        result['cache_decode_verified'] = True
        observed = result['content_audit']['cache_inventory_sha256']
        if args.expected_cache_digest and observed != args.expected_cache_digest:
            raise ValueError('whole-cache inventory SHA mismatch')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
