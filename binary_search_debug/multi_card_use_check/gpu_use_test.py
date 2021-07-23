import os

os.system("ln -sf /work/paddle/build/python/build/lib.linux-x86_64-3.5/paddle /usr/local/lib/python3.5/dist-packages")
res = os.popen("python /work/scripts/multi_card_use_check/base_test.py").read()
lines = res.splitlines()
if lines[-1] == "True":
    print("chenweihang: commit build & test: use multiple gpu cards.")
else:
    print("chenweihang: commit build & test: use signal gpu card.")