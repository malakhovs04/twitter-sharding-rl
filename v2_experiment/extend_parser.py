# coding: utf-8
from pathlib import Path
import json,hashlib
old=Path('v2/prepare.py').read_text()
s=old.replace('first!=0','not (0<=first<900)')
s=s.replace("if op=='get':a[1]+=1", "if op in ('get','gets'):a[1]+=1")
s=s.replace("elif op=='set':a[2]+=1;a[4]+=ks+vs", "elif op in ('set','add'):a[2]+=1")
s=s.replace("if op=='set':\n                version", "if op=='set' or (op=='add' and k not in live):\n                a[4]+=ks+vs\n                version")
s=s.replace("'get':1,'set':2", "'get_or_gets':1,'set_or_add_attempt':2")
s=s.replace('TTL request semantics assume successful writes; no eviction.', 'set assumed successful, add conditional on reconstructed absence, gets read-only; TTL relative seconds, no eviction or success flags.')
Path('prepare_offset.py').write_text(s)
p=Path('v2_results/offset_supplement_protocol.json');d=json.loads(p.read_text());d.update(clusters=[10,45],change='Accept first event within first slot. gets read-only work1; add attempt work2, update inventory and dirty bytes only when key absent in reconstructed state. No model/selection changes.',excluded_cluster45='Original refusal preserved; now evaluated as explicitly revised parser scenario before any policy performance for this cluster',supplement_parser_sha256=hashlib.sha256(Path('prepare_offset.py').read_bytes()).hexdigest(),semantic_source='https://docs.memcached.org/protocols/basic/',assumptions='No success flags; empty reconstructed cache, no eviction. Conditional add semantics are scenario, not actual measured outcomes. TTL interpreted as relative trace TTL.');p.write_text(json.dumps(d,indent=2)+'\n')
