"""Key-level aggregates, exact UInt64 mapping; no inspection of policy performance."""
from pathlib import Path
from collections import defaultdict,Counter
import csv,subprocess,heapq,json,hashlib
import numpy as np
from data_pipeline import key_hash,jch

def prepare(source,output):
    source=Path(source);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    ids={};events=[defaultdict(lambda:np.zeros(5,dtype=np.float64)) for _ in range(672)]
    snapshots=[];live={};expiry=[];version=0;boundary=0;last=-1;ops=Counter();unknown=Counter();first=None
    proc=subprocess.Popen(['zstd','-dc',str(source)],stdout=subprocess.PIPE,text=True)
    def expire(t):
        while expiry and expiry[0][0]<=t:
            _,v,k=heapq.heappop(expiry)
            if k in live and live[k][1]==v:del live[k]
    try:
        for row in csv.reader(proc.stdout):
            if len(row)!=7:raise ValueError('Expected seven trace fields')
            t,key,ks,vs,_,op,ttl=row;t=int(t);ks=int(ks);vs=int(vs);ttl=int(ttl)
            if t<last or min(t,ks,vs,ttl)<0:raise ValueError('Unordered or negative fields')
            if first is None:first=t
            last=t
            if t>=604800:continue
            while boundary*900<=t:
                expire(boundary*900);snapshots.append({k:x[0] for k,x in live.items()});boundary+=1
            expire(t)
            if key not in ids:ids[key]=len(ids)
            k=ids[key];a=events[t//900][k];a[0]+=1;ops[op]+=1
            # Strict get/set support; unknown operations retained in counts then rejected.
            if op=='get':a[1]+=1
            elif op=='set':a[2]+=1;a[4]+=ks+vs
            else:unknown[op]+=1
            a[3]+=ks+vs
            if op=='set':
                version+=1;live[k]=(ks+vs,version)
                if ttl:heapq.heappush(expiry,(t+ttl,version,k))
    finally:
        if proc.stdout:proc.stdout.close()
        status=proc.wait()
    if status or first!=0 or last<604799:raise ValueError('Incomplete or invalid trace')
    if unknown:raise ValueError(f'Unsupported operations, no silent semantic substitution: {dict(unknown)}')
    while boundary<=672:
        expire(boundary*900);snapshots.append({k:x[0] for k,x in live.items()});boundary+=1
    n=len(ids)
    # float64 avoids rounding exact request totals; bytes remain int64 in inventory.
    work=np.zeros((672,n));counts=np.zeros((672,n),dtype=np.int32);writes=np.zeros((672,n))
    for t,ev in enumerate(events):
        for k,a in ev.items():
            counts[t,k]=int(a[0]);work[t,k]=a[1]+2*a[2]+a[3]/4096
            # Every write to a migrating key adds its reported full bytes to a conservative dirty log.
            writes[t,k]=a[4]
    inventory=np.zeros((673,n),dtype=np.int64)
    for t,snap in enumerate(snapshots):
        for k,size in snap.items():inventory[t,k]=size
    hashes=np.array([key_hash(k) for k in ids],dtype=np.uint64)
    mapping=np.array([[jch(int(h),n) for h in hashes] for n in range(2,9)],dtype=np.int8)
    for name,array in dict(work=work,counts=counts,writes=writes,inventory=inventory,mapping=mapping,hashes=hashes).items():np.save(output/(name+'.npy'),array)
    meta=dict(source=str(source.name),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),keys=n,requests=int(counts.sum()),operations=dict(ops),slots=672,slot_seconds=900,weights={'get':1,'set':2,'reported_bytes_divisor':4096},status='Scenario work units, not measured CPU/latency. TTL request semantics assume successful writes; no eviction.')
    (output/'metadata.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)

def load(path):
    return {n:np.load(Path(path)/(n+'.npy'),mmap_mode='r') for n in ('work','counts','writes','inventory','mapping','hashes')}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');a=p.parse_args();prepare(a.source,a.output)
