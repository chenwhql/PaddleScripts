import re

_first_cap_re = re.compile('(.)([A-Z][a-z]+)')
_all_cap_re = re.compile('([a-z])([A-Z])')


def _convert_camel_to_snake(name):
    s1 = _first_cap_re.sub(r'\1_\2', name)
    print(s1) # Simple_ImgConv_Pool
    return _all_cap_re.sub(r'\1_\2', s1).lower()

print(_convert_camel_to_snake("SimpleImgConvPool"))
# simple_img_conv_pool
