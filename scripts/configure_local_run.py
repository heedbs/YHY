"""Prepare a path-adapted local config; does not train or load a model."""
import argparse,hashlib,json,os
from pathlib import Path
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['stage1','stage2'],required=True);ap.add_argument('--template',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 if a.output.exists(): raise RuntimeError('output exists')
 root=Path(__file__).resolve().parents[1]/a.stage
 c=json.loads(a.template.read_text());split=Path(os.environ['ORCHID_MANIFEST_DIR'])/'final_outer_inner_split_manifest.csv'
 if sha(split)!=c['split_sha256']: raise RuntimeError('frozen split SHA mismatch; do not regenerate or substitute split')
 files=[root/'train_one_run.py',*sorted((root/'orchid').glob('*.py'))]
 if a.stage=='stage1': files += [root/'validate_before_run.py',root/'evaluate_oof.py']
 h=hashlib.sha256()
 for p in sorted(files):h.update(str(p.relative_to(root)).encode());h.update(sha(p).encode())
 c['code_sha256']=h.hexdigest()
 with a.output.open('x') as f:json.dump(c,f,indent=2)
 print('Local reproduction config written; historical_code_sha256 remains recorded.')
if __name__=='__main__':main()
