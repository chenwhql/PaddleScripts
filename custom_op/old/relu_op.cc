// relu_op.cc
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

// 前向OP的输入X、输出Y、属性
class Relu2OpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensor.");
    AddOutput("Y", "Output of relu_op");
    AddComment(R"DOC(
Relu Operator.
Y = max(X, 0)
)DOC");
  }
};

// 前向OP的定义和InferShape实现，设置输出Y的shape
class Relu2Op : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto in_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim("Y", in_dims);
  }
};

// 实现前向OP的Kernel计算函数: Y = max(0, X)
using Tensor = framework::Tensor;
template <typename DeviceContext, typename T>
class Relu2Kernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in_t = ctx.Input<Tensor>("X");
    auto* out_t = ctx.Output<Tensor>("Y");
    auto x = in_t->data<T>();
    // mutable_data分配内存、获取指针
    auto y = out_t->mutable_data<T>(ctx.GetPlace());
    for (int i = 0; i < in_t->numel(); ++i) {
      y[i] = std::max(static_cast<T>(0.), x[i]);
    }
  }
};

// 定义反向OP的输入Y和dY、输出dX、属性:
template <typename T>
class Relu2GradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> op) const override {
    op->SetType("relu2_grad");
    op->SetInput("Y", this->Output("Y"));
    op->SetInput(framework::GradVarName("Y"), this->OutputGrad("Y"));
    op->SetAttrMap(this->Attrs());
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

// 定义反向OP和InferShape实现,设置dX的shape
class Relu2GradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto in_dims = ctx->GetInputDim(framework::GradVarName("Y"));
    ctx->SetOutputDim(framework::GradVarName("X"), in_dims);
  }
};

// 实现反向OP的kernel函数 dx = dy * ( y > 0. ? 1. : 0)
template <typename DeviceContext, typename T>
class Relu2GradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* dy_t = ctx.Input<Tensor>(framework::GradVarName("Y"));
    auto* y_t = ctx.Input<Tensor>("Y");
    auto* dx_t = ctx.Output<Tensor>(framework::GradVarName("X"));

    auto dy = dy_t->data<T>();
    auto y = y_t->data<T>();
    auto dx = dx_t->mutable_data<T>(ctx.GetPlace());

    for (int i = 0; i < y_t->numel(); ++i) {
      dx[i] = dy[i] * (y[i] > static_cast<T>(0) ? 1. : 0.);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = paddle::platform::CPUDeviceContext;
// 注册前向和反向op
// 为了和框架内部的relu区分，这里注册的OP type为relu2
REGISTER_OPERATOR(relu2,
                  ops::Relu2Op,
                  ops::Relu2OpMaker,
                  ops::Relu2GradMaker<paddle::framework::OpDesc>,
                  ops::Relu2GradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(relu2_grad, ops::Relu2GradOp);
// 注册CPU的Kernel
REGISTER_OP_CPU_KERNEL(relu2,
                       ops::Relu2Kernel<CPU, float>,
                       ops::Relu2Kernel<CPU, double>);
REGISTER_OP_CPU_KERNEL(relu2_grad,
                       ops::Relu2GradKernel<CPU, float>,
                       ops::Relu2GradKernel<CPU, double>);
