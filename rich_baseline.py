"""Non-neural baseline with the same causal counterfactual-routing observations."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from simulator import Config,load_data,SPLITS,summarize
from stronger_agent import RichEnv,rank

def evaluate(data,c,split,margin,mode):
    env=RichEnv(data,c,*SPLITS[split]);s=env.reset();done=False;rows=[]
    while not done:
        n=env.n;last=s[10:17]
        predicted=np.maximum(last,s[17:24]) if mode=='max2' else last
        costs=[]
        for action in (-1,0,1):
            new=min(8,max(2,n+action));p=predicted[new-2]
            migration=s[45+new-2] # features 45:52 = current migration row.
            costs.append((int(p>margin),max(0,float(p-margin)),new+4*migration))
        action=min(range(3),key=lambda i:costs[i])-1
        s,_,done,row=env.step(action);rows.append(row)
    return pd.DataFrame(rows)

def select(root,out):
    root=Path(root);out=Path(out);data=load_data(root/'data/processed.npz');c=Config(**json.loads((root/'results/config.json').read_text()));rows=[]
    for margin in (.7,.85,1.):
        for mode in ('last','max2'):
            trace=evaluate(data,c,'validation',margin,mode);rows.append(dict(margin=margin,mode=mode,**summarize(trace)))
    pd.DataFrame(rows).to_csv(out/'rich_baseline_validation.csv',index=False)
    selected=min(rows,key=lambda r:(r['overloaded_slots'],r['excess_requests'],r['cost']))
    (out/'rich_baseline_selection.json').write_text(json.dumps(selected,indent=2));print(selected)
