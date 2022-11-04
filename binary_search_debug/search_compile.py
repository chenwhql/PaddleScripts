import os

def get_candidates(git_command=''):
    init_command = 'git pull upstream develop'
    os.system(init_command)
    git_command = 'git log --pretty=format:"%h" --since=2021-03-02 --until=2021-04-07'
    res = os.popen(git_command).read()
    candidates = res.split('\n')
    for line in candidates:
        print(line)
    return candidates


def compile(commit):
    cmake = 'cmake ../.. -DPY_VERSION=3.7 -DCMAKE_BUILD_TYPE=Release -DWITH_GPU=ON -DCUDA_ARCH_NAME=Auto -DWITH_TESTING=OFF -DWITH_DISTRIBUTE=ON'
    command = 'git reset --hard %s' % commit
    os.system(command)
    build = 'build_versions/build_%s' % commit
    print(build)
    if not os.path.exists(build):
        os.system("mkdir %s" % build)
    command = 'cd %s && %s && make -j24' %(build, cmake)
    os.system(command)


def softlink(commit):
    build = 'build_versions/build_%s' % commit
    os.system("ln -sf /work/dev1/paddle/%s/python/build/lib.linux-x86_64-3.7/paddle /work/dev1/paddle/build_venv/lib/python3.7/site-packages" % build)


def condition(commit):
    print(commit)
    compile(commit)
    softlink(commit)
    res = os.popen("python /work/scripts/binary_search_debug/seg_fault/queue_seg.py").read()
    lines = res.splitlines()
    if lines[-1] == "Success":
        print("Chen Weihang: commit build & test: run success.")
        return False
    else:
        print("Chen Weihang: commit build & test: segmentation fault.")
        return True


def binary_search(candidates, cond):
    left = 0
    right = len(candidates)
    while left < right:
        mid = left + (right - left)//2
        commit = candidates[mid]

        if cond(commit):
            left = mid + 1
        else:
            right = mid
    return left


def iterable_search(candidates, cond):
    for commit in candidates:
        if cond(commit):
            break


if __name__ == "__main__":
    candidates = get_candidates('')
    binary_search(candidates, condition)
    # iterable_search(candidates, condition)