"""Prepare local paths from an authorized exact manifest bundle; no model execution.

Generated files contain linked study identifiers and local paths. Keep them private.
This tool does not download data, generate cache, decode images, or load tensors.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def safe_source(root, relative):
    name = PurePosixPath(relative)
    if name.is_absolute() or '..' in name.parts or relative != name.as_posix() or not name.parts:
        raise ValueError('unsafe source-relative path')
    result = (root / Path(*name.parts)).resolve()
    if root.resolve() not in result.parents:
        raise ValueError('source path escapes local source root')
    return result


def read_rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def validate_rows(patches, splits, expected):
    if len(patches) != expected['patches'] or not patches or not splits:
        raise ValueError('patch/partition record count mismatch')
    patients = {}; sources = {}; uids = set()
    for row in patches:
        uid = row['uid']
        if not uid or Path(uid).name != uid or '\\' in uid or uid in {'.', '..'} or uid in uids:
            raise ValueError('invalid or duplicate patch UID')
        uids.add(uid)
        patient, target = row['patient_key'], row['cls']
        if not patient or target not in {'normal', 'osmf', 'oscc'}:
            raise ValueError('invalid patient/target')
        if patients.setdefault(patient, target) != target:
            raise ValueError('patient target contradiction')
        source = row['source_sha256']
        if sources.setdefault(source, (patient, target, row['source_relative_path'])) != (patient, target, row['source_relative_path']):
            raise ValueError('source identity/target contradiction')
        if any(int(row[k]) < 0 for k in ('x', 'y')) or any(int(row[k]) <= 0 for k in ('w', 'h')):
            raise ValueError('invalid crop bounds')
    if len(patients) != expected['patients'] or len(sources) != expected['sources']:
        raise ValueError('patient/source count mismatch')
    if {c: list(patients.values()).count(c) for c in ('normal', 'osmf', 'oscc')} != expected['class_patients']:
        raise ValueError('patient class counts mismatch')
    groups = {}; seen = set(); test_counts = {p: 0 for p in patients}
    for row in splits:
        fold = int(row['outer_fold']); part = row['split']; patient = row['patient_id']
        if fold not in range(5) or part not in {'train', 'inner_val', 'outer_test'} or patient not in patients:
            raise ValueError('unexpected split identity')
        key = (fold, patient)
        if key in seen or row['class'] != patients[patient]:
            raise ValueError('duplicate split membership or wrong target')
        seen.add(key); groups.setdefault((fold, part), set()).add(patient)
        if part == 'outer_test': test_counts[patient] += 1
    for fold in range(5):
        parts = [groups.get((fold, p), set()) for p in ('train', 'inner_val', 'outer_test')]
        if any(not part for part in parts) or set.union(*parts) != set(patients):
            raise ValueError('incomplete fold partitions')
        if any(parts[i] & parts[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise ValueError('patient overlap')
    if any(count != 1 for count in test_counts.values()):
        raise ValueError('outer-test membership must be exactly once')


def prepare(bundle, source_root, cache_root, weights_map, output):
    if output.exists() or output.is_symlink():
        raise ValueError('output collision; never overwrite local manifests')
    identity = json.loads((ROOT / 'data/manifest_bundle_identity.json').read_text())
    for name, expected in identity['files'].items():
        file = bundle / name
        if file.is_symlink() or sha(file) != expected:
            raise ValueError('authorized bundle SHA mismatch')
    patches = read_rows(bundle / 'patches.csv')
    split_path = bundle / 'final_outer_inner_split_manifest.csv'
    splits = read_rows(split_path)
    validate_rows(patches, splits, identity['expected'])
    split_sha = sha(split_path)
    for stage in ('stage1', 'stage2'):
        for template in (ROOT / 'configs' / stage).glob('*.json'):
            if json.loads(template.read_text())['split_sha256'] != split_sha:
                raise ValueError('bundle split differs from historical config identity')
    output_rows = []
    for row in patches:
        source = safe_source(source_root, row['source_relative_path'])
        cache = (cache_root / (row['uid'] + '.jpg')).resolve()
        if not source.is_file() or not cache.is_file():
            raise ValueError('source image or pre-existing cache missing; no cache is generated')
        output_rows.append({k: row[k] for k in ('uid', 'x', 'y', 'w', 'h', 'cls', 'patient_key', 'source_sha256')})
        output_rows[-1].update(path=str(source), cache_path=str(cache))
    weights = json.loads(weights_map.read_text())
    provenance = read_rows(ROOT / 'data/pretrained_sources.csv')
    for row in provenance:
        path = Path(weights[row['model_name']]).expanduser().resolve()
        if not path.is_file() or sha(path) != row['weight_sha256']:
            raise ValueError('pretrained file identity mismatch')
        row['local_cache_path'] = str(path)
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    def write_csv(name, rows):
        with (output / name).open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(rows)
    write_csv('final_unique_patch_manifest.csv', output_rows)
    # Preserve exact split bytes: config/checkpoint split identity is not rebound.
    with (output / split_path.name).open('xb') as handle:
        handle.write(split_path.read_bytes())
    write_csv('selected_model_pretrained_provenance.csv', provenance)
    receipt = {'status': 'LOCAL_INPUT_BINDING_ONLY_NOT_MODEL_EXECUTION',
               'patches': len(patches), 'patients': identity['expected']['patients'],
               'sources': identity['expected']['sources'], 'split_sha256': split_sha,
               'source_content_rehashed': False, 'cache_content_rehashed': False,
               'model_initialization_verified': False,
               'files': {f.name: sha(f) for f in output.iterdir() if f.is_file()}}
    with (output / 'LOCAL_BINDING_RECEIPT.json').open('x') as handle:
        json.dump(receipt, handle, indent=2)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'source-root', 'cache-root', 'weights-map', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    receipt = prepare(args.bundle, args.source_root.resolve(), args.cache_root.resolve(), args.weights_map, args.output)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
