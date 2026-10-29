"""Streaming equivalent of prepare_offset + routing-signature compaction.
No dense (time, every-key) arrays: at most 128 routing signatures for n=2..8.
"""
from pathlib import Path
from collections import Counter,defaultdict
import subprocess,csv,heapq,json,hashlib
import numpy as np
from data_pipeline import key_hash,jch

def prepare(source,output):
    source=Path(source);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    values={n:np.zeros((673 if n=='inventory' else 672,128)) for n in ('work','counts','writes','inventory')}
    signatures={};groups={};live={};expiry=[];stock=np.zeros(128);hot=np.zeros(672);keywork=defaultdict(float)
    boundary=0;version=0;last=-1;first=None;ops=Counter();unknown=Counter();rows=0
    def expire(t):
        while expiry and expiry[0][0]<=t:
            _,v,k=heapq.heappop(expiry)
            if k in live and live[k][1]==v:
                size,_,g=live.pop(k);stock[g]-=size
    p=subprocess.Popen(['zstd','-dc',str(source)],stdout=subprocess.PIPE,text=True)
    try:
        for row in csv.reader(p.stdout):
            if len(row)!=7:raise ValueError('Expected seven fields')
            t,k,ks,vs,_,op,ttl=row;t=int(t);ks=int(ks);vs=int(vs);ttl=int(ttl)
            if t<last or min(t,ks,vs,ttl)<0:raise ValueError('Unordered or negative fields')
            if first is None:first=t
            last=t
            if t>=604800:continue
            while boundary*900<=t:
                if boundary:
                    hot[boundary-1]=max(keywork.values(),default=0.);keywork.clear()
                expire(boundary*900);values['inventory'][boundary]=stock;boundary+=1
            expire(t)
            if k not in groups:
                h=key_hash(k);sig=tuple(jch(h,n) for n in range(2,9))
                if sig not in signatures:signatures[sig]=len(signatures)
                groups[k]=signatures[sig]
            g=groups[k];slot=t//900;ops[op]+=1;rows+=1
            if op in ('get','gets'):units=1
            elif op in ('set','add'):units=2
            else:unknown[op]+=1;units=0
            w=units+(ks+vs)/4096;values['work'][slot,g]+=w;keywork[k]+=w;values['counts'][slot,g]+=1
            if op=='set' or (op=='add' and k not in live):
                if k in live:
                    size,_,oldg=live[k];stock[oldg]-=size
                version+=1;live[k]=(ks+vs,version,g);stock[g]+=ks+vs;values['writes'][slot,g]+=ks+vs
                if ttl:heapq.heappush(expiry,(t+ttl,version,k))
    finally:
        if p.stdout:p.stdout.close()
        status=p.wait()
    if status or first is None or not 0<=first<900 or last<604799:raise ValueError('Coverage failure')
    if unknown:raise ValueError('Unsupported operations '+str(dict(unknown)))
    while boundary<=672:
        if boundary:hot[boundary-1]=max(keywork.values(),default=0.);keywork.clear()
        expire(boundary*900);values['inventory'][boundary]=stock;boundary+=1
    n=len(signatures)
    for name,a in values.items():np.save(output/(name+'.npy'),a[:,:n])
    np.save(output/'mapping.npy',np.array(list(signatures),dtype=np.int8).T);np.save(output/'hot_work.npy',hot)
    meta=dict(source=source.name,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),keys=len(groups),routing_groups=n,requests=rows,operations=dict(ops),first_second=first,last_second=last,semantics='get/gets read-only work1; set/add attempts work2; add writes only if absent in reconstructed state; relative TTL, no eviction/success flags')
    (output/'metadata.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');a=p.parse_args();prepare(a.source,a.output)
