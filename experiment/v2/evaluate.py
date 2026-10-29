"""One-shot evaluation of prespecified unseen clusters, after hash freeze."""
from pathlib import Path
from dataclasses import asdict,replace
import json,hashlib,urllib.request,time
import pandas as pd
import numpy as np
from agents import MLP
from .prepare import prepare,load
from .environment import calibrate,rollout,summary
from .train import load_table,table_policy,neural_policy,threshold,target_policy

ROOT=Path('v2_results')

def policies(selection,all_seeds=False):
    result={}
    for kind in ('q','dqn_matched','dqn_continuous'):
        x=selection[kind];seeds=(7,42,123) if all_seeds else (7,)
        for seed in seeds:
            path=ROOT/f'{kind}_{x["candidate"]}_s{seed}.npz'
            result[f'{kind}_seed{seed}']=table_policy(load_table(path)) if kind=='q' else neural_policy(MLP.load(path),kind=='dqn_matched')
    b=selection['threshold'];result['threshold']=threshold(b['up'],b['down'])
    result['selected_fixed_target']=target_policy(int(selection['fixed_target']['target']))
    return result

def main():
    frozen=json.loads((ROOT/'frozen.json').read_text())
    for name,digest in frozen['files'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Frozen file changed: '+name)
    report=ROOT/'holdout';report.mkdir(exist_ok=True)
    if (report/'opened.json').exists():raise RuntimeError('Holdout already opened; do not silently rerun or retune')
    (report/'opened.json').write_text(json.dumps(dict(time_unix=time.time(),frozen_sha256=hashlib.sha256((ROOT/'frozen.json').read_bytes()).hexdigest(),retuning_permitted=False),indent=2))
    protocol=json.loads(Path('v2/protocol.json').read_text());selection=json.loads((ROOT/'selection.json').read_text());pool=policies(selection,True)
    rows=[];failures=[];diagnostics=[];capacity_rows=[]
    for cluster in protocol['holdout_clusters']:
        source=Path(f'data/cluster{cluster}.sort.sample100.zst');folder=Path(f'data/cluster{cluster}')
        try:
            if not source.exists():
                url='https://ftp.pdl.cmu.edu/pub/datasets/twemcacheWorkload/open_source/sample100/'+source.name
                tmp=source.with_suffix('.partial');urllib.request.urlretrieve(url,tmp);tmp.replace(source)
            prepare(source,folder);d=load(folder);c=calibrate(d,0,96)
        except Exception as e:
            failures.append(dict(cluster=cluster,error=str(e)));(report/'failures.json').write_text(json.dumps(failures,indent=2));print('CLUSTER FAILED',cluster,str(e),flush=True);continue
        (report/f'cluster{cluster}_config.json').write_text(json.dumps(asdict(c),indent=2))
        # Raw hot-key floor distinguished from collisions: key alone > per-server work capacity.
        for day in range(1,7):
            w=d['work'][day*96:(day+1)*96]
            diagnostics.append(dict(cluster=cluster,day=day+1,hot_key_over_capacity_slots=int((w.max(1)>c.capacity).sum()),mean_work=float(w.sum(1).mean()),mean_inventory_mb=float(d['inventory'][day*96:(day+1)*96].sum(1).mean()/1e6)))
        for name,policy in pool.items():
            trace=rollout(d,c,96,672,policy);pd.DataFrame(trace).to_csv(report/f'cluster{cluster}_{name}.csv',index=False)
            rows.append(dict(cluster=cluster,strategy=name,**summary(trace)))
        # Prespecified robustness: frozen policies, capacity and bandwidth separately varied; no selection.
        for factor in (.5,2):
            for parameter in ('capacity','bandwidth'):
                changed=replace(c,**{parameter:getattr(c,parameter)*factor})
                for name,policy in policies(selection).items():
                    capacity_rows.append(dict(cluster=cluster,parameter=parameter,factor=factor,strategy=name,**summary(rollout(d,changed,96,672,policy))))
        pd.DataFrame(rows).to_csv(report/'comparison.csv',index=False);pd.DataFrame(diagnostics).to_csv(report/'daily_diagnostics.csv',index=False)
        pd.DataFrame(capacity_rows).to_csv(report/'sensitivity.csv',index=False)
        print('EVALUATED',cluster,pd.DataFrame(rows)[lambda f:f.cluster==cluster].to_string(index=False),flush=True)
    (report/'failures.json').write_text(json.dumps(failures,indent=2))
    (report/'complete.json').write_text(json.dumps(dict(requested_clusters=protocol['holdout_clusters'],evaluated=sorted({r['cluster'] for r in rows}),failures=failures,retuned=False),indent=2))

if __name__=='__main__':main()
