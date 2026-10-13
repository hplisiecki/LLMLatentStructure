"""Check supplied input integrity against the release manifest."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'validation/input_manifest.json').read_text())
for relative,expected in manifest.items():
    p=root/relative
    assert p.is_file(),relative
    assert hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==expected['sha256'],relative
    assert p.stat().st_size==expected['bytes'],relative
print(f'Input integrity verified: {len(manifest)} files')
