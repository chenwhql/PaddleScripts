# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import unittest
import paddle
import numpy as np
from paddle.utils.cpp_extension import load

# Compile and load custom op Just-In-Time.
custom_module = load(
    name='custom_zty',
    sources=['test_zty_custom.cc'],
    verbose=True)

class TestJITLoad(unittest.TestCase):
    def test_api(self):
        raw_data = np.array([[-1, 1, 0], [1, -1, -1]]).astype('float32')
        x = paddle.to_tensor(raw_data, dtype='float32')
        # use custom api
        out = custom_module.custom_zty(x)
        print(out)


if __name__ == '__main__':
    unittest.main()
