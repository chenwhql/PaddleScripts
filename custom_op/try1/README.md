# 自定义Op工作拆解和形态

## 1. 写法简化

- 用户仅编写如下组件
  - 前向计算函数
  - 反向计算函数
  - 注册Op

- 方案实验
  - 用function参数分析技术解析用户定义func的输入和返回类型
    - 根据对func的解析构建OpProto和基础组件
  - 封装function到自定义OpKernel类中
    - 构建OpKernel
    - 需要升级paddle至C++14编译（完全兼容，成本很低）
  - 声明更简介的多数据类型注册宏

参考当前目录示例

## 2. 编译与载入简化

原编译流程：

1. 用户使用裸编译指令编译得到.so

```
# PaddlePaddel >=1.6.1, 仅需要include ${include_dir} 和 ${include_dir}/third_party
nvcc relu_op.cu -c -o relu_op.cu.o -ccbin cc -DPADDLE_WITH_CUDA -DEIGEN_USE_GPU -DPADDLE_WITH_MKLDNN -Xcompiler -fPIC -std=c++11 -Xcompiler -fPIC -w --expt-relaxed-constexpr -O3 -DNVCC \
    -I ${include_dir} \
    -I ${include_dir}/third_party \

g++ relu_op.cc relu_op.cu.o -o relu2_op.so -shared -fPIC -std=c++11 -O3 -DPADDLE_WITH_MKLDNN \
  -I ${include_dir} \
  -I ${include_dir}/third_party \
  -L /usr/local/cuda/lib64 \
  -L ${lib_dir} -lpaddle_framework -lcudart
```

2. 调用paddle api载入.so

```
# 调用load_op_library加载动态库
fluid.load_op_library('relu2_op.so')
```

3. 封装python API

```
def relu2(x, name=None):
    # relu2的type和在OP中定义的type相同
    helper = LayerHelper("relu2", **locals())
    # 创建输出Variable
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(type="relu2", inputs={"X": [x]}, outputs={"Y": out})
    return out
```

4. import使用

新调用流程（期望）：

1. setup编译

- 简单易懂的配置，一键执行

```
from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='lltm_cpp',
      ext_modules=[cpp_extension.CppExtension('lltm_cpp', ['lltm.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
```

2. import使用

实现方法（待进行）
- 参考torch.utils.cpp_extension的封装
- 和我们编译做的事情类似，只是封装得友好，目前判断可行
- 需要了解setuptools的一些实现和功能，了解编译选项等

## 3. API暴露设计与简化

三条路线：
- 对于需要的概念，全部进行二次封装（Tensor, dtype, place, register_op）
- 部分二次封装，Tensor复用预测的封装
- 长远而谨慎的设计，paddle要暴露哪些C++ API，现在先暴露哪些，高层讨论