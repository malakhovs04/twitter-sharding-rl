from pathlib import Path
from dataclasses import asdict,replace
import hashlib,json,time
import pandas as pd
from prepare_offset import prepare
from v2.prepare import load
from v2.environment import calibrate,rollout,summary
from v2.evaluate import policies
r=Path('v2_results');h=r/'holdout'
if (h/'offset_supplement_opened.json').exists():raise RuntimeError('Supplement already opened')
for name,digest in json.loads((r/'frozen.json').read_text())['files'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
protocol=json.loads((r/'offset_supplement_protocol.json').read_text())
assert hashlib.sha256(Path('prepare_offset.py').read_bytes()).hexdigest()==protocol['supplement_parser_sha256']
(h/'offset_supplement_opened.json').write_text(json.dumps(dict(time_unix=time.time(),protocol_sha256=hashlib.sha256((r/'offset_supplement_protocol.json').read_bytes()).hexdigest()),indent=2))
selection=json.loads((r/'selection.json').read_text());rows=[];diags=[];sensitivity=[];failures=[]
for cluster in (10,45):
    try:
        prepare(f'data/cluster{cluster}.sort.sample100.zst',f'data/cluster{cluster}')
        d=load(f'data/cluster{cluster}');c=calibrate(d,0,96)
    except Exception as e:
        failures.append(dict(cluster=cluster,error=str(e)));continue
    (h/f'cluster{cluster}_config.json').write_text(json.dumps(asdict(c),indent=2))
    for name,policy in policies(selection,True).items():
        trace=rollout(d,c,96,672,policy);pd.DataFrame(trace).to_csv(h/f'cluster{cluster}_{name}.csv',index=False)
        rows.append(dict(cluster=cluster,strategy=name,**summary(trace)))
    for day in range(1,7):
        w=d['work'][day*96:(day+1)*96]
        diags.append(dict(cluster=cluster,day=day+1,hot_key_over_capacity_slots=int((w.max(1)>c.capacity).sum()),mean_work=float(w.sum(1).mean()),mean_inventory_mb=float(d['inventory'][day*96:(day+1)*96].sum(1).mean()/1e6)))
    for factor in (.5,2):
        for parameter in ('capacity','bandwidth'):
            changed=replace(c,**{parameter:getattr(c,parameter)*factor})
            for name,policy in policies(selection).items():sensitivity.append(dict(cluster=cluster,parameter=parameter,factor=factor,strategy=name,**summary(rollout(d,changed,96,672,policy))))
    pd.DataFrame(rows).to_csv(h/'offset_supplement_comparison.csv',index=False)
    pd.DataFrame(diags).to_csv(h/'offset_supplement_diagnostics.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(h/'offset_supplement_sensitivity.csv',index=False)
    print(pd.DataFrame(rows)[lambda f:f.cluster==cluster].to_string(index=False),flush=True)
(h/'offset_supplement_complete.json').write_text(json.dumps(dict(clusters=sorted({row['cluster'] for row in rows}),model_and_selection_unchanged=True,parser_change_only=True,failures=failures),indent=2))
