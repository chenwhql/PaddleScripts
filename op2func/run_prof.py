import os

for i in range(50):
    res = os.popen("python /work/scripts/op2func/slice_perf.py").read()
    lines = res.splitlines()
    if lines[-1] == "Success":
        break
