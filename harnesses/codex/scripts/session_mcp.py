"""Fresh MCP connections for this run's metadata calls; never starts GPU jobs."""
import asyncio
import json
import os
from pathlib import Path
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {'get_task_brief', 'search_stats', 'list_iterations', 'get_iteration',
           'get_lineage', 'get_memory', 'add_memory', 'record_summary',
           'get_token_usage', 'scf_check_eligibility'}

async def calls(requests):
    assert all(r['tool'] in ALLOWED for r in requests)
    params = StdioServerParameters(command=sys.executable,
        args=[str(ROOT/'mcp_server/server.py')], cwd=str(ROOT),
        env={**os.environ, 'DFTSR_ALLOW_TEST':'0',
             'DFTSR_SCF_RUNS_DIR':os.environ.get('DFTSR_SCF_RUNS_DIR',str(ROOT/'runs/scf'))})
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            output = []
            for r in requests:
                result = await session.call_tool(r['tool'], r.get('args', {}))
                value = ''.join(c.text for c in result.content if c.type == 'text')
                if getattr(result, 'isError', False):
                    raise RuntimeError(f"MCP {r['tool']} failed: {value}")
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass
                output.append(value)
            return output

if __name__ == '__main__':
    requests = json.load(sys.stdin)
    print(json.dumps(asyncio.run(calls(requests)), indent=2))
