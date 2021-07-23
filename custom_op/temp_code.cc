using KernelFunc = std::vector<Tensor> (*)(std::vector<Tensor> inputs,
                                           std::vector<boost::any> attrs);
template <typename T>
struct TypeTag {};
template <typename F, F f>
struct KernelFuncImpl;
template <typename Return, typename... Args, Return (*impl_fn)(Args...)>
struct KernelFuncImpl<Return (*)(Args...), impl_fn> {
  static Return Compute(std::vector<Tensor> inputs,
                        std::vector<boost::any> attrs) {
    return ComputeCallHelper<Args..., TypeTag<int>>::template Compute<0, 0>(
        inputs, attrs);
  }
 private:
  template <typename... RemainingArgs>
  struct ComputeCallHelper;
  // for Tensor input
  template <typename... Tail>
  struct ComputeCallHelper<const Tensor&, Tail...> {
    template <int in_idx, int attr_idx, typename... PreviousArgs>
    static Return Compute(std::vector<Tensor> inputs,
                          std::vector<boost::any> attrs,
                          const PreviousArgs&... pargs) {
      static_assert(attr_idx == 0,
                    "Input tensor should appear before attributes.");
      const Tensor& arg = inputs[in_idx];
      return ComputeCallHelper<Tail...>::template Compute<in_idx + 1, attr_idx>(
          inputs, attrs, pargs..., arg);
    }
  };
  // TODO(chenweihang): add support for attribute input
  // int attribute input (not used now)
  template <typename... Tail>
  struct ComputeCallHelper<int, Tail...> {
    template <int in_idx, int attr_idx, typename... PreviousArgs>
    static Return Compute(std::vector<Tensor> inputs,
                          std::vector<boost::any> attrs,
                          const PreviousArgs&... pargs) {
      try {
        int arg = boost::any_cast<int>(attrs[attr_idx]);
        return ComputeCallHelper<Tail...>::template Compute<in_idx,
                                                            attr_idx + 1>(
            inputs, attrs, pargs..., arg);
      } catch (boost::bad_any_cast&) {
        throw std::runtime_error(
            "Attribute cast error in custom operator. Expected int value.");
      }
    }
  };
  // end: base template
  template <typename T>
  struct ComputeCallHelper<TypeTag<T>> {
    template <int in_idx, int attr_idx>
    static Return Compute(std::vector<Tensor> inputs,
                          std::vector<boost::any> attrs, const Args&... args) {
      return impl_fn(args...);
    }
  };
};
#define PD_KERNEL(...) \
  ::paddle::KernelFuncImpl<decltype(&__VA_ARGS__), &__VA_ARGS__>::Compute