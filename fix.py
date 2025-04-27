# fix_cli_config.py
import yaml, sys, collections
src, dst = sys.argv[1], sys.argv[2]
with open(src) as f:
    flat = yaml.safe_load(f)

nested = {}
for k, v in flat.items():
    cur = nested
    *parents, last = k.split(".")
    for p in parents:
        cur = cur.setdefault(p, {})
    cur[last] = v

with open(dst, "w") as f:
    yaml.safe_dump(nested, f)
