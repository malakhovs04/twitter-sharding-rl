"""Resume interrupted evaluation with equivalent routing-group representation.
No changes to models, calibration, selection, scenarios or metrics.
"""
from pathlib import Path
from dataclasses import asdict,replace
import hashlib,json,shutil
import numpy as np
import pandas as pd
from compact_data import CompactEnv,load_compact
from v2.environment import calibrate,summary
from v2.evaluate import policies
r=Path('v2_results');h=r/'holdout'
if (h/'offset_supplement_complete.json').exists():raise RuntimeError('Completed already')
for name,digest in json.loads((r/'frozen.json').read_text())['files'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
(h/'compact_resume.json').write_text(json.dumps(dict(reason='Dense execution interrupted during sensitivity, not model selection. Equivalent grouping tested; existing primary traces compared numerically before replacement.',files={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['compact_data.py','prepare_compact.py','test_compact.py','test_streaming_compact.py','resume_compact_evaluation.py']},retuned=False),indent=2))
selection=json.loads((r/'selection.json').read_text());rows=[];diags=[];sensitivity=[];checked=[]
def rollout(d,c,policy):
    e=CompactEnv(d,c,96,672);s=e.reset();done=False;result=[]
    while not done:s,_,done,row=e.step(int(policy(s)));result.append(row)
    return result
for cluster in (10,45):
    d=load_compact(f'data/compact_cluster{cluster}');c=calibrate(d,0,96)
    (h/f'cluster{cluster}_config.json').write_text(json.dumps(asdict(c),indent=2))
    for name,policy in policies(selection,True).items():
        trace=rollout(d,c,policy);frame=pd.DataFrame(trace);path=h/f'cluster{cluster}_{name}.csv'
        if path.exists():
            old=pd.read_csv(path);np.testing.assert_allclose(frame.to_numpy(),old.to_numpy(),rtol=1e-9,atol=1e-6)
            backup=h/'dense_partial';backup.mkdir(exist_ok=True);shutil.copy2(path,backup/path.name);checked.append(path.name)
        frame.to_csv(path,index=False);rows.append(dict(cluster=cluster,strategy=name,**summary(trace)))
    for day in range(1,7):
        w=d['work'][day*96:(day+1)*96]
        diags.append(dict(cluster=cluster,day=day+1,hot_key_over_capacity_slots=int((d['hot_work'][day*96:(day+1)*96]>c.capacity).sum()),mean_work=float(w.sum(1).mean()),mean_inventory_mb=float(d['inventory'][day*96:(day+1)*96].sum(1).mean()/1e6)))
    for factor in (.5,2):
        for parameter in ('capacity','bandwidth'):
            changed=replace(c,**{parameter:getattr(c,parameter)*factor})
            for name,policy in policies(selection).items():sensitivity.append(dict(cluster=cluster,parameter=parameter,factor=factor,strategy=name,**summary(rollout(d,changed,policy))))
    print(pd.DataFrame(rows)[lambda f:f.cluster==cluster].to_string(index=False),flush=True)
pd.DataFrame(rows).to_csv(h/'offset_supplement_comparison.csv',index=False)
pd.DataFrame(diags).to_csv(h/'offset_supplement_diagnostics.csv',index=False)
pd.DataFrame(sensitivity).to_csv(h/'offset_supplement_sensitivity.csv',index=False)
(h/'offset_supplement_complete.json').write_text(json.dumps(dict(clusters=[10,45],model_and_selection_unchanged=True,parser_changes_documented=True,compact_equivalence_checked=checked,failures=[]),indent=2))
