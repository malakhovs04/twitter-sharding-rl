"""Stream official Twitter sample; request routing and write-defined live inventory."""
from pathlib import Path
from collections import Counter
import subprocess,csv,hashlib,heapq,json
import numpy as np
NODES=tuple(range(2,9)); SLOT=900
SOURCE='https://ftp.pdl.cmu.edu/pub/datasets/twemcacheWorkload/open_source/sample100/cluster7.sort.sample100.zst'
def jch(key,n):
    if type(key) is not int or not 0<=key<2**64 or type(n) is not int or n<1:raise ValueError('UInt64 / buckets')
    b,j=-1,0
    while j<n:
        b=j;key=(key*2862933555777941757+1)&((1<<64)-1);j=((b+1)*(1<<31))//((key>>33)+1)
    return b

def key_hash(key):return int.from_bytes(hashlib.blake2b(key.encode(),digest_size=8,person=b'twitter1').digest(),'little')

class Inventory:
    """Only successful semantics are assumed for write-like trace requests; not actual cache occupancy."""
    def __init__(self):
        self.objects={};self.heap=[];self.version=0;self.matrix=np.zeros((7,7),dtype=np.int64);self.bytes=0
    def _change(self,key,size,mapping,sign):
        self.bytes+=sign*size
        a=np.asarray(mapping);self.matrix+=(a[:,None]!=a[None,:])*(sign*size)
    def expire(self,t):
        while self.heap and self.heap[0][0]<=t:
            expiry,version,key=heapq.heappop(self.heap)
            old=self.objects.get(key)
            if old and old[2]==version:self.remove(key)
    def remove(self,key):
        old=self.objects.pop(key,None)
        if old:self._change(key,old[0],old[3],-1)
    def write(self,key,size,ttl,t,mapping):
        self.remove(key);self.version+=1
        expiry=t+ttl if ttl else float('inf')
        self.objects[key]=(size,expiry,self.version,mapping);self._change(key,size,mapping,1)
        if ttl:heapq.heappush(self.heap,(expiry,self.version,key))

def prepare(path='data/cluster7.sort.sample100.zst',output='data/processed.npz'):
    path=Path(path);sha=hashlib.sha256(path.read_bytes()).hexdigest()
    # fixed grid; requires at least a full 168-hour trace, no repeated/stitched events.
    loads=np.zeros((672,7,8),dtype=np.int64);migrations=np.zeros((673,7,7),dtype=np.int64)
    live=np.zeros(673,dtype=np.int64);unique=Counter();ops=Counter();ttl_hist=Counter();sizes=Counter()
    inv=Inventory();cache={};boundary=0;last=-1;rows=0;first=None;bytes_requested=0;zero=0;source_rows=0;source_last=-1;excluded=0
    proc=subprocess.Popen(['zstd','-dc',str(path)],stdout=subprocess.PIPE,text=True,encoding='utf8')
    try:
        for row in csv.reader(proc.stdout):
            if len(row)!=7:raise ValueError('Unexpected trace schema')
            t,k,ks,vs,client,op,ttl=row;t=int(t);ks=int(ks);vs=int(vs);ttl=int(ttl)
            if min(t,ks,vs,ttl)<0 or t<last:raise ValueError('Negative field or unordered trace')
            if first is None:first=t
            source_rows+=1;source_last=t
            if t>=672*SLOT:
                excluded+=1;last=t;continue
            # Capture inventory at boundary BEFORE this slot's requests.
            while boundary*SLOT<=t:
                inv.expire(boundary*SLOT);migrations[boundary]=inv.matrix;live[boundary]=inv.bytes;boundary+=1
            inv.expire(t)
            if k not in cache:cache[k]=tuple(jch(key_hash(k),n) for n in NODES)
            mapping=cache[k];h=t//SLOT
            for i,s in enumerate(mapping):loads[h,i,s]+=1
            if op in ('set','add','replace','cas','append','prepend'):
                # Request success is unknown; assume last write wins and reported full size.
                inv.write(k,ks+vs,ttl,t,mapping);ttl_hist[ttl]+=1
            elif op=='delete':inv.remove(k)
            unique[k]+=1;ops[op]+=1;sizes[vs]+=1;zero+=vs==0;bytes_requested+=ks+vs
            rows+=1;last=t
        if proc.wait()!=0:raise RuntimeError('zstd decompression failed')
    finally:
        if proc.poll() is None:proc.terminate();proc.wait()
    while boundary<=672:
        inv.expire(boundary*SLOT);migrations[boundary]=inv.matrix;live[boundary]=inv.bytes;boundary+=1
    if first!=0 or last<167*3600:raise ValueError(f'Insufficient coverage: {first}..{last}')
    assert np.all(loads.sum(2)==loads[:,0].sum(1)[:,None])
    assert loads[:,0].sum()==rows
    Path(output).parent.mkdir(exist_ok=True)
    np.savez_compressed(output,loads=loads,migrations=migrations,live=live)
    total=sum(unique.values());top=unique.most_common(20)
    meta=dict(source=SOURCE,sha256=sha,rows=rows,unique_keys=len(unique),first_second=first,last_second=source_last,selected_last_second=min(source_last,604799),source_rows=source_rows,excluded_tail_rows=excluded,slot_seconds=SLOT,hours=168,operations=dict(ops),zero_value_rows=zero,request_key_value_bytes=bytes_requested,ttl_write_hist=dict(ttl_hist),top20_request_fraction=sum(v for _,v in top)/total,sampling='Server-provided sample100; selection algorithm unverified; do not extrapolate to production',inventory='Write-defined, empty initial inventory; last write wins; TTL relative seconds, 0=no expiry. Reads do not create objects. No cache eviction or success flags.')
    Path(output).with_suffix('.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps({k:meta[k] for k in ('rows','unique_keys','first_second','last_second','operations','top20_request_fraction')},indent=2),flush=True)
    return loads,migrations,live
if __name__=='__main__':prepare()
