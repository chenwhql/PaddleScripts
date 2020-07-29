import paddle

def version_compare(version1, version2):
    version1 = version1.split(".")
    version2 = version2.split(".")
    num = min(len(version1), len(version2))
    for index in range(num):
        try:
            vn1 = int(version1[index])
        except:
            vn1 = 0
        try:
            vn2 = int(version2[index])
        except:
            vn2 = 0

        if vn1 > vn2:
            return True
        elif vn1 < vn2:
            return False
    return len(version1) > len(version2)

if version_compare(paddle.__version__, "1.8.0"):
    print("verison greater than 1.8")
    import paddle.fluid as fluid
else:
    print("verison is: %s" % paddle.__version__)
    import paddle.fluid.dygraph as dygraph

dygraph.enable_dygraph()
