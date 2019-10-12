#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <armadillo>
#include <map>
#include <string>

namespace ozcode {
class Optimizer {
 public:
  Optimizer() : learning_rate_(0.01) {}
  Optimizer(double lr) : learning_rate_(lr) {}
  virtual ~Optimizer() = default;

  /**
   * \brief Update params using grads
   * \param params
   * \param grads
   */
  virtual void Update(std::map<std::string, arma::mat>& params,
                      std::map<std::string, arma::mat> const& grads) = 0;

 protected:
  double learning_rate_;
};
}  // namespace ozcode

#endif
