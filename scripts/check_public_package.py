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
 findings=[];files=[p for p in root.rglob('*') if p.is_file() or p.is_symlink()]
 forbidden={'.pt','.pth','.safetensors','.npz','.png','.jpg','.jpeg','.tif','.tiff','.docx'}
 for p in files:
  rel=str(p.relative_to(root))
  if '.git' in p.parts or p.name.startswith('.env') or p.suffix.lower() in forbidden:findings.append({'file':rel,'rule':'forbidden_file_type','matches':1})
  if p.is_symlink():
   findings.append({'file':rel,'rule':'symlink','matches':1})
   continue
  if p.stat().st_size>5*1024*1024:
   findings.append({'file':rel,'rule':'file_above_5MiB','matches':1})
   continue
  value=p.read_text(errors='replace')
  for name,pattern in RULES.items():
   matches=list(re.finditer(pattern,value))
   if matches:findings.append({'file':rel,'rule':name,'matches':len(matches)})
 return {'files_scanned':len(files),'findings':findings,'pass':not findings}

def verify_manifest(root):
 findings=[];seen=set();checked=0
 manifest=root/'SHA256_MANIFEST.txt'
 if manifest.is_symlink() or not manifest.is_file():
  return {'findings':[{'file':'SHA256_MANIFEST.txt','rule':'manifest_missing_or_symlink','matches':1}],'manifest_files_checked':0}
 for line_number,line in enumerate(manifest.read_text().splitlines(),1):
  fields=line.split(maxsplit=1)
  if len(fields)!=2 or not re.fullmatch(r'[0-9a-f]{64}',fields[0]):
   findings.append({'file':'SHA256_MANIFEST.txt','rule':'malformed_entry','line':line_number,'matches':1});continue
  expected,name=fields;relative=Path(name)
  if relative.is_absolute() or '..' in relative.parts or name!=relative.as_posix() or name=='SHA256_MANIFEST.txt':
   findings.append({'file':'SHA256_MANIFEST.txt','rule':'unsafe_manifest_path','line':line_number,'matches':1});continue
  if name in seen:
   findings.append({'file':name,'rule':'duplicate_manifest_entry','matches':1});continue
  seen.add(name);p=root/relative
  if any(parent.is_symlink() for parent in [p,*p.parents] if parent!=root and root in parent.parents):
   findings.append({'file':name,'rule':'manifest_symlink','matches':1});continue
  if not p.is_file():
   findings.append({'file':name,'rule':'manifest_file_missing','matches':1});continue
  digest=hashlib.sha256()
  with p.open('rb') as handle:
   for chunk in iter(lambda:handle.read(1024*1024),b''):digest.update(chunk)
  if digest.hexdigest()!=expected:findings.append({'file':name,'rule':'sha_mismatch','matches':1})
  checked+=1
 actual={str(p.relative_to(root)) for p in root.rglob('*') if (p.is_file() or p.is_symlink()) and p!=manifest}
 for name in sorted(actual-seen):findings.append({'file':name,'rule':'unlisted_file','matches':1})
 return {'findings':findings,'manifest_files_checked':checked}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('--manifest',action='store_true');a=ap.parse_args();result=scan(a.root)
 if a.manifest:
  verification=verify_manifest(a.root)
  result['findings'].extend(verification['findings'])
  result['manifest_files_checked']=verification['manifest_files_checked']
 result['pass']=not result['findings'];print(json.dumps(result,indent=2));raise SystemExit(0 if result['pass'] else 1)
if __name__=='__main__':main()
