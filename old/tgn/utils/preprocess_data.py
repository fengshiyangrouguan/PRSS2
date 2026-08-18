import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse


def preprocess(data_name):
  u_list, i_list, ts_list, label_list = [], [], [], []
  feat_l = []
  idx_list = []

  with open(data_name) as f:
    s = next(f)
    for idx, line in enumerate(f):
      e = line.strip().split(',')
      u = int(e[0])
      i = int(e[1])

      ts = float(e[2])
      label = float(e[3])  # int(e[3])

      feat = np.array([float(x) for x in e[4:]])

      u_list.append(u)
      i_list.append(i)
      ts_list.append(ts)
      label_list.append(label)
      idx_list.append(idx)

      feat_l.append(feat)
  return pd.DataFrame({'u': u_list,
                       'i': i_list,
                       'ts': ts_list,
                       'label': label_list,
                       'idx': idx_list}), np.array(feat_l)


def reindex(df, bipartite=True):
  new_df = df.copy()
  if bipartite:
    assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
    assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

    upper_u = df.u.max() + 1
    new_i = df.i + upper_u

    new_df.i = new_i
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1
  else:
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1

  return new_df


def run(data_name, bipartite=True, input_path=None, output_dir="data"):
  """Preprocess a JODIE CSV without requiring a duplicate inside ``data/``.

  ``input_path`` is intentionally optional so the original TGN command keeps working.
  The Wikipedia file is large, therefore the diagnostics pipeline points this function at
  the user-provided CSV instead of copying roughly 560 MB first.
  """
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  PATH = str(Path(input_path)) if input_path is not None else str(output_dir / '{}.csv'.format(data_name))
  OUT_DF = str(output_dir / 'ml_{}.csv'.format(data_name))
  OUT_FEAT = str(output_dir / 'ml_{}.npy'.format(data_name))
  OUT_NODE_FEAT = str(output_dir / 'ml_{}_node.npy'.format(data_name))

  df, feat = preprocess(PATH)
  new_df = reindex(df, bipartite)

  empty = np.zeros(feat.shape[1])[np.newaxis, :]
  feat = np.vstack([empty, feat])

  max_idx = max(new_df.u.max(), new_df.i.max())
  # Match the real edge-feature width instead of silently assuming Wikipedia's 172.
  rand_feat = np.zeros((max_idx + 1, feat.shape[1]), dtype=np.float32)

  new_df.to_csv(OUT_DF)
  np.save(OUT_FEAT, feat)
  np.save(OUT_NODE_FEAT, rand_feat)

parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
parser.add_argument('--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bipartite', action='store_true', help='Whether the graph is bipartite')
parser.add_argument('--input', type=str, default=None,
                    help='Path to the raw CSV (defaults to <output_dir>/<data>.csv)')
parser.add_argument('--output_dir', type=str, default='data',
                    help='Directory for ml_<data>.csv/.npy files')

args = parser.parse_args()

run(args.data, bipartite=args.bipartite, input_path=args.input, output_dir=args.output_dir)
