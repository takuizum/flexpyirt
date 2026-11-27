# ruff: noqa
import json, sys
import numpy as np

a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
print("| case | iter base/new | |Δloglik| | max |Δestimate| | max rel |ΔSE| |")
print("|---|---|---|---|---|")
for k in a:
    x, y = a[k], b[k]
    de = np.max(np.abs(np.array(x["estimate"]) - np.array(y["estimate"])))
    if x["se"] is not None and not all(v != v for v in x["se"]):
        s1, s2 = np.array(x["se"], float), np.array(y["se"], float)
        m = np.isfinite(s1)
        ds = f"{np.max(np.abs(s1[m] - s2[m]) / np.abs(s1[m])):.1e}"
    else:
        ds = "–"
    print(
        f"| {k} | {x['n_iter']}/{y['n_iter']} | {abs(x['loglik'] - y['loglik']):.1e} | {de:.1e} | {ds} |"
    )
