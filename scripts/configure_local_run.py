"""Bind an existing scientific config to relocated code; never train or load weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_stage(stage, config):
    expected = {
        'stage1': (80, {'InceptionV3', 'EfficientNet-Lite2', 'MobileViT-S', 'EfficientNet-Lite2+Mamba'},
                   'inner_patient_macro_ovr_auroc|inner_patient_macro_f1|validation_loss|earlier_epoch'),
        'stage2': (40, {'Composite-Mamba', 'Composite-NoMamba'},
                   'inner_patient_macro_f1|macro_ovr_auroc|validation_loss|earlier_epoch'),
    }
    epochs, models, criterion = expected[stage]
    if config.get('model_name') not in models or config.get('epochs') != epochs:
        raise RuntimeError('stage/template model or epoch mismatch')
    if config.get('checkpoint_criterion') != criterion:
        raise RuntimeError('stage/template checkpoint criterion mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['stage1', 'stage2'], required=True)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError('output exists or is a symlink')
    config = json.loads(args.template.read_text())
    validate_stage(args.stage, config)
    manifest_dir = os.environ.get('ORCHID_MANIFEST_DIR')
    if not manifest_dir:
        raise RuntimeError('set ORCHID_MANIFEST_DIR to the approved manifest directory')
    split = Path(manifest_dir) / 'final_outer_inner_split_manifest.csv'
    if sha(split) != config['split_sha256']:
        raise RuntimeError('frozen split SHA mismatch; do not regenerate or substitute split')
    root = Path(__file__).resolve().parents[1] / args.stage
    files = [root / 'train_one_run.py', *sorted((root / 'orchid').glob('*.py'))]
    if args.stage == 'stage1':
        files += [root / 'validate_before_run.py', root / 'evaluate_oof.py']
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(sha(path).encode())
    config['code_sha256'] = digest.hexdigest()
    with args.output.open('x') as handle:
        json.dump(config, handle, indent=2)
    print('Local reproduction config written; historical_code_sha256 remains recorded.')


if __name__ == '__main__':
    main()
