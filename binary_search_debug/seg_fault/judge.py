import os

# os.system("ln -sf /work/paddle/build/python/build/lib.linux-x86_64-3.5/paddle /usr/local/lib/python3.5/dist-packages")
res = os.popen("python /work/scripts/binary_search_debug/seg_fault/queue_seg.py").read()
lines = res.splitlines()
if lines[-1] == "Success":
    print("Chen Weihang: commit build & test: run success.")
else:
    print("Chen Weihang: commit build & test: segmentation fault.")