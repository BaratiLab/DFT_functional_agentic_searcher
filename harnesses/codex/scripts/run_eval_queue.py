"""Coordinate three concurrent calls to the existing evaluation MCP server.

Candidates and plans are supplied by agents. This runner does not design forms,
fit outside MCP, change scores, or access held-out results.
"""
import asyncio
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT/'runs/eval_queue'
TARGET_OK = 400      # stop after this many non-seed iterations with status 'ok'
MAX_RECORDS = 480    # ... or after this many non-seed records in total (ok + error + running)


def counts():
    con=sqlite3.connect(f'file:{ROOT}/runs/iterations.sqlite?mode=ro',uri=True)
    rows=con.execute('SELECT status,COUNT(*) FROM iterations WHERE id>3 GROUP BY status').fetchall()
    con.close()
    return dict(rows)


async def run():
    params=StdioServerParameters(command=sys.executable,
        args=[str(ROOT/'mcp_server/server.py')],cwd=str(ROOT),
        env={**os.environ,'DFTSR_GPUS':os.environ.get('DFTSR_GPUS','0'),'DFTSR_ALLOW_TEST':'0',
             'DFTSR_JOB_TIMEOUT':'560','PYTHONWARNINGS':'ignore',
             'DFTSR_SCF_RUNS_DIR':os.environ.get('DFTSR_SCF_RUNS_DIR',str(ROOT/'runs/scf'))})
    async with stdio_client(params) as (read,write):
        async with ClientSession(read,write) as session:
            await session.initialize()
            print('Shared MCP client initialized',flush=True)
            # Coordinator-owned warm-up, on the sole GPU-owning connection.
            warm = await session.call_tool('warm_up_workers', {'wait': False})
            print('WARMUP '+''.join(c.text for c in warm.content if c.type=='text'),flush=True)
            active=set()
            async def evaluate(path):
                args=json.loads(path.read_text())
                print(f'SUBMIT {path.stem} {args["name"]}',flush=True)
                try:
                    response=await session.call_tool('evaluate_functional',args)
                    result=json.loads(''.join(c.text for c in response.content if c.type=='text'))
                except Exception as exc:
                    result={'infrastructure_error':repr(exc)}
                target=QUEUE/'results'/path.name
                temporary=target.with_suffix('.tmp')
                temporary.write_text(json.dumps(result,indent=2)+'\n')
                temporary.replace(target)
                print(json.dumps({'key':path.stem,'id':result.get('iteration_id'),'status':result.get('status'),
                    'val':result.get('val_wrmsd'),'violations':result.get('n_violations'),
                    'error':result.get('infrastructure_error',result.get('error'))}),flush=True)
            while True:
                active={task for task in active if not task.done()}
                state=counts()
                done=state.get('ok',0)>=TARGET_OK or sum(state.values())>=MAX_RECORDS
                # Coordinator advances this gate after the independent audit.
                gate_path=QUEUE/'checkpoint_limit'
                gate=int(gate_path.read_text()) if gate_path.exists() else 50
                (QUEUE/'coordinator_status.json').write_text(json.dumps({'time':time.time(),'counts':state,'active':len(active)}))
                if done and not active:
                    break
                if not (QUEUE/'pause').exists():
                    for source in sorted((QUEUE/'pending').glob('*.json'))[:3-len(active)]:
                        state=counts()
                        if state.get('ok',0)+len(active)>=min(TARGET_OK,gate) or sum(state.values())>=MAX_RECORDS:
                            break
                        target=QUEUE/'active'/source.name
                        source.replace(target)
                        active.add(asyncio.create_task(evaluate(target)))
                        # Let the server record the submission before checking the count again.
                        await asyncio.sleep(.2)
                await asyncio.sleep(2)
            print(f'COMPLETE: {counts()}',flush=True)


if __name__=='__main__':
    asyncio.run(run())
