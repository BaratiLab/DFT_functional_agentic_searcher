#!/usr/bin/env python3
"""CPU synthetic-density checks of every frozen functional and coefficient shape."""
import json,torch
from runtime import ROOT,configure
configure()
from dftsr.api import load_candidate,params_to_tensors
from dftsr.worker import smoke_test
from dftsr.xc_torch import Density,C_TAU
n=32;rho=torch.logspace(-6,1,n,dtype=torch.float64);b=.4*rho
sig=.2*rho**(8/3);sigb=.1*b**(8/3)
d=Density(rho,b,sig,torch.zeros_like(sig),sigb,C_TAU*rho**(5/3)+sig/(8*rho),C_TAU*b**(5/3)+sigb/(8*b))
for f in sorted((ROOT/'functionals').glob('*.py')):
    func=load_candidate(str(f));params=json.loads(f.with_suffix('.params.json').read_text())
    init=func.init_params();assert params.keys()==init.keys(),f.name
    assert all(len(params[k])==len(init[k]) for k in params),f.name
    out=func.energy_density(params_to_tensors(params,'cpu'),d)
    assert out.shape==(n,) and torch.isfinite(out).all(),f.name
    smoke_test(func);print('PASS',f.stem)
