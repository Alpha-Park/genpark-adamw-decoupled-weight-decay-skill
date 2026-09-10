import json
import sys
from client import AdamWOptimizer

opt = AdamWOptimizer()

def handle_rpc(line):
    try:
        req = json.loads(line)
        method = req.get("method")
        params = req.get("params", {})
        rid = req.get("id")
        
        if method == "tools/list":
            tools = [
                {"name": "adamw_step", "description": "Perform single optimization step on scalar parameter"}
            ]
            return json.dumps({"jsonrpc": "2.0", "id": rid, "result": {"tools": tools}})
        elif method == "tools/call":
            tname = params.get("name")
            args = params.get("arguments", {})
            if tname == "adamw_step":
                p = opt.step(args["param"], args["grad"], args.get("param_id", 0))
                return json.dumps({"jsonrpc": "2.0", "id": rid, "result": {"new_param": p}})
    except Exception as e:
        return json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32603, "message": str(e)}})

if __name__ == "__main__":
    for line in sys.stdin:
        if line.strip():
            print(handle_rpc(line.strip()), flush=True)
