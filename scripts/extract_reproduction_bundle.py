"""Extract two exact metadata CSVs outside the repository; no images or models."""
import argparse
import hashlib
import json
from pathlib import Path
from release_data import load_policy, read_reviewed

ROOT = Path(__file__).resolve().parents[1]


def extract(output):
    if output.exists() or output.is_symlink(): raise ValueError('output collision')
    if output.resolve().is_relative_to(ROOT): raise ValueError('keep extracted local inputs outside repository')
    policy = load_policy(ROOT)
    identity = json.loads((ROOT / 'data/manifest_bundle_identity.json').read_text())
    payloads = {}
    for name, path in [('patches.csv', 'data/reproduction/patches.csv.gz'),
                       ('final_outer_inner_split_manifest.csv', 'data/reproduction/final_outer_inner_split_manifest.csv')]:
        payload = read_reviewed(ROOT, policy[path])
        if hashlib.sha256(payload).hexdigest() != identity['files'][name]: raise ValueError('historical bundle SHA mismatch')
        payloads[name] = payload
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    for name, payload in payloads.items():
        with (output / name).open('xb') as handle: handle.write(payload)
    return {'status':'EXACT_METADATA_BUNDLE_EXTRACTED_NO_MODEL_EXECUTION', 'files':list(payloads)}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True)
    print(json.dumps(extract(parser.parse_args().output),indent=2))
