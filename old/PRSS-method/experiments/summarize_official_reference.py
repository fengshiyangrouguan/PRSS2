#!/usr/bin/env python3
import argparse, json, pickle
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--workdir',required=True); p.add_argument('--output',required=True); p.add_argument('--prefix',required=True)
a=p.parse_args(); wd=Path(a.workdir)
rp=wd/'results'/f'{a.prefix}_node_classification.pkl'
if not rp.exists(): raise FileNotFoundError(rp)
d=pickle.load(open(rp,'rb'))
summary={
  'mode':'official_reference_exact_copy',
  'source':'twitter-research/tgn train_supervised.py (unmodified copy)',
  'official_reported_test_auc': float(d.get('test_ap')),
  'test': {'auc': float(d.get('test_ap')), 'ap': None, 'nll': None},
  'note':'Official script names this field test_ap but eval_node_classification returns ROC-AUC. Exact upstream protocol has no separate held-out validation unless --use_validation is passed.'
}
out=Path(a.output); out.mkdir(parents=True,exist_ok=True); json.dump(summary,open(out/'summary.json','w'),indent=2); json.dump({'status':'complete'},open(out/'_SUCCESS.json','w'),indent=2)
print(json.dumps(summary,indent=2))
