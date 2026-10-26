"""Read-only local-input verification. No model imports, image decode or GPU calls."""
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
    args = parser.parse_args()
    env = os.environ.get('ORCHID_MANIFEST_DIR')
    if not env or Path(env).resolve() != args.manifest_dir.resolve():
        raise ValueError('ORCHID_MANIFEST_DIR must match --manifest-dir')
    print(json.dumps(check(args.stage, args.config, args.manifest_dir, args.bundle, args.output), indent=2))


if __name__ == '__main__':
    main()
