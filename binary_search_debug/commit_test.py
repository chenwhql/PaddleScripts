import os
import sys

commit = sys.argv[1]
print("compile test for commit: %s" % commit)

# reset
# command = 'git reset --hard %s' % commit
# os.system(command)

# build
# build = 'build_versions/build_%s' % commit
# cmake = 'cmake ../.. -DPY_VERSION=3.7 -DCMAKE_BUILD_TYPE=Release -DWITH_GPU=ON -DCUDA_ARCH_NAME=Auto -DWITH_TESTING=OFF -DWITH_DISTRIBUTE=ON'
# if not os.path.exists(build):
#     os.system("mkdir %s" % build)
# command = 'cd %s && %s && make -j24' %(build, cmake)
# os.system(command)

# link
build = 'build_versions/build_%s' % commit
os.system("ln -sf /work/dev1/paddle/%s/python/build/lib.linux-x86_64-3.7/paddle /work/dev1/paddle/build_venv/lib/python3.7/site-packages" % build)

# test
res = os.popen("python /work/scripts/binary_search_debug/seg_fault/queue_seg.py").read()
lines = res.splitlines()
if lines[-1] == "Success":
    print("Chen Weihang: commit build & test: run success.")
else:
    print("Chen Weihang: commit build & test: segmentation fault.")
