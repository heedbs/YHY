"""Read-only heuristic scan and optional SHA verification; prints no matched secret values."""
import argparse,hashlib,json,re
from pathlib import Path

RULES={
 'private_absolute_path':r'/(?:home|Users|root|opt/anaconda3)/[A-Za-z0-9_.-]+',
 'source_patient_identifier':r'\bo-[0-4]-\d{2}-\d{2}(?:-\d+)?\b',
 'email':r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
 'private_key':r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
 'github_token':r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b',
 'cloud_access_key':r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
 'credential_assignment':r'''(?i)(?:api_key|access_token|secret_key|password)\s*[:=]\s*["'][A-Za-z0-9_+/=-]{12,}["']''',
}
def scan(root):
 findings=[];files=[p for p in root.rglob('*') if p.is_file()]
 forbidden={'.pt','.pth','.safetensors','.npz','.png','.jpg','.jpeg','.tif','.tiff','.docx'}
 for p in files:
  rel=str(p.relative_to(root))
  if '.git' in p.parts or p.name.startswith('.env') or p.suffix.lower() in forbidden:findings.append({'file':rel,'rule':'forbidden_file_type','matches':1})
  if p.is_symlink():findings.append({'file':rel,'rule':'symlink','matches':1})
  if p.stat().st_size>5*1024*1024:findings.append({'file':rel,'rule':'file_above_5MiB','matches':1})
  value=p.read_text(errors='replace')
  for name,pattern in RULES.items():
   matches=list(re.finditer(pattern,value))
   if matches:findings.append({'file':rel,'rule':name,'matches':len(matches)})
 return {'files_scanned':len(files),'findings':findings,'pass':not findings}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('--manifest',action='store_true');a=ap.parse_args();result=scan(a.root)
 if a.manifest:
  checked=0
  for line in (a.root/'SHA256_MANIFEST.txt').read_text().splitlines():
   expected,name=line.split(maxsplit=1);p=a.root/name
   if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:result['findings'].append({'file':name,'rule':'sha_mismatch','matches':1})
   checked+=1
  result['manifest_files_checked']=checked
 result['pass']=not result['findings'];print(json.dumps(result,indent=2));raise SystemExit(0 if result['pass'] else 1)
if __name__=='__main__':main()
