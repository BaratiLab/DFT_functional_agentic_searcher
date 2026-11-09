"""Paths and global settings (override with environment variables)."""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get('DFTSR_DATA_ROOT', os.path.join(PROJECT_ROOT, '..', '..', 'work', 'data'))
ACCDB_DIR = os.path.join(DATA_ROOT, 'external', 'ACCDB')
MGCDB84_DIR = os.path.join(ACCDB_DIR, 'Databases', 'MGCDB84')

# ωB97M-V/def2-QZVPPD densities (PySCF grid level 3, SG-1 VV10 grid) used for non-SCF evaluation
DENSITY_TAG = os.environ.get('DFTSR_DENSITY_TAG', 'wb97mv_def2qzvppd_L3')
DENSITY_DIR = os.path.join(DATA_ROOT, 'densities', DENSITY_TAG)
CONSTRAINT_DIR = os.path.join(DATA_ROOT, 'densities', 'constraints')
FEATURE_CACHE_DIR = os.path.join(DATA_ROOT, 'feature_cache', DENSITY_TAG)

# Search bookkeeping lives in the project (small, version-controllable)
RUNS_DIR = os.environ.get('DFTSR_RUNS_DIR', os.path.join(PROJECT_ROOT, 'runs'))
DB_PATH = os.path.join(RUNS_DIR, 'iterations.sqlite')
SCF_RUNS_DIR = os.environ.get('DFTSR_SCF_RUNS_DIR', os.path.join(DATA_ROOT, 'scf_runs_codex'))

BASIS = 'def2-qzvppd'
GRID_LEVEL = 3
OMEGA = 0.3
