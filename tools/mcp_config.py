#!/usr/bin/env python3
"""Print configuration only; does not edit any client settings."""
import argparse,json,sys
from runtime import ROOT
p=argparse.ArgumentParser();p.add_argument('client',choices=['claude','codex']);a=p.parse_args()
h='opus' if a.client=='claude' else 'codex'
args=[str(ROOT/'tools/harness.py'),h,'mcp']
env={'DFTSR_GPUS':'0','DFTSR_ALLOW_TEST':'0','DFTSR_PREWARM':'0','DFTSR_JOB_TIMEOUT':'560','DFTSR_FIT_BUDGET':'410'}
if a.client=='claude':print(json.dumps({'mcpServers':{'dft-functional-search':{'command':sys.executable,'args':args,'env':env}}},indent=2))
else:
    print('[mcp_servers.dft-functional-search]\ncommand = '+json.dumps(sys.executable))
    print('args = '+json.dumps(args)+'\nstartup_timeout_sec = 60\ntool_timeout_sec = 600\nenabled = true')
    print('\n[mcp_servers.dft-functional-search.env]')
    for k,v in env.items():print(k+' = '+json.dumps(v))
