#!/usr/bin/env python3
"""Discover both MCP tool lists without warming GPUs or calling a calculation."""
import asyncio,os,sys,json
from runtime import ROOT
from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client
async def check(name):
    env={**os.environ,'DFTSR_PREWARM':'0','DFTSR_ALLOW_TEST':'0','PYTHONDONTWRITEBYTECODE':'1',
         'DFTSR_RUNS_DIR':str(ROOT/'work/checks'/name)}
    p=StdioServerParameters(command=sys.executable,args=[str(ROOT/'tools/harness.py'),name,'mcp'],env=env)
    async with stdio_client(p) as (r,w):
        async with ClientSession(r,w) as s:
            await s.initialize();listing=await s.list_tools();names={t.name for t in listing.tools}
            assert {'evaluate_functional','get_task_brief','scf_submit','final_test_evaluation'}<=names
            if name=='opus':assert {'gscdb_scf_submit','syfes_submit'}<=names
            print(name,len(names),'tools discovered')
async def main():
    for name in ['opus','codex']:await asyncio.wait_for(check(name),60)
if __name__=='__main__':asyncio.run(main())
