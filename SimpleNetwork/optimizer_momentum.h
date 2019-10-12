#ifndef OPTIMIZER_MOMENTUM_H
#define OPTIMIZER_MOMENTUM_H

#include "optimizer.h"

namespace ozcode {
class optimizer_momentum : Optimizer {
 public:
  optimizer_momentum() : Optimizer(0.01) {}
  optimizer_momentum(double learning_rate, double momentum)
      : Optimizer(learning_rate), momentum_(momentum) {}
  void Update(std::map<std::string, arma::mat>& params,
              std::map<std::string, arma::mat> const& grads) override;

 private:
  double momentum_ = 0.9;  // 类的私有成员采用变量名后跟下划线的方式命名
};
}  // namespace ozcode

#endif
