#!/usr/bin/env python3
"""Launch one packaged script or MCP server with portable paths."""
import argparse,os,sys
from runtime import configure
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('harness',choices=['opus','codex']);p.add_argument('script',help='mcp or a filename under scripts/')
p.add_argument('arguments',nargs=argparse.REMAINDER)
a=p.parse_args();rest=a.arguments;h=configure(a.harness)
f=h/'mcp_server/server.py' if a.script=='mcp' else h/'scripts'/a.script
if not f.is_file():p.error('script not found')
os.chdir(h);os.execv(sys.executable,[sys.executable,str(f),*rest])
