"""Shared portable paths. Does not allocate a GPU or modify existing source trees."""
from pathlib import Path
import os, sys
ROOT=Path(__file__).resolve().parents[1]
def configure(harness='opus'):
    h=ROOT/'harnesses'/harness
    defaults={'DFTSR_DATA_ROOT':ROOT/'work/data','GSCDB_WORK':ROOT/'work/gscdb',
              'GSCDB_ROOT':ROOT/'work/gscdb/GSCDB','GSCDB_REF_DIR':ROOT/'work/gscdb/reference',
              'GSCDB_RUNS':ROOT/'work/gscdb/runs','DFTSR_SCF_RUNS_DIR':ROOT/'work'/harness/'scf',
              'DFTSR_ALLOW_TEST':'0','DFTSR_PREWARM':'0','DFTSR_GPUS':'0',
              'PYTHONDONTWRITEBYTECODE':'1'}
    for k,v in defaults.items():os.environ.setdefault(k,str(v))
    sys.path.insert(0,str(h))
    return h
