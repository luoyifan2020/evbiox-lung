# ========================
# preprocess_clinical.py
# ========================
import pandas as pd
from pathlib import Path

ROOT     = Path(__file__).parent.parent
META_CSV = ROOT / 'data' / 'interim' / 'lidc_preproc' / 'Meta' / 'meta_info.csv'
OUT_CSV  = ROOT / 'data' / 'clinical' / 'clinical.csv'
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# 读取并聚合 patient-level 临床特征
df = pd.read_csv(META_CSV)
# 选取所有数值型字段（除 patient_id），采用中位数汇总
num_cols = [c for c in df.select_dtypes(include=['number']).columns if c != 'patient_id']
agg_dict = {c: 'median' for c in num_cols}
features = df.groupby('patient_id').agg(agg_dict).reset_index()

features.to_csv(OUT_CSV, index=False)
print('临床表型已写入', OUT_CSV)
