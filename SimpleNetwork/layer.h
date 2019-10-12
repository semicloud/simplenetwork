#ifndef LAYER_H
#define LAYER_H

#include <armadillo>
#include <string>

namespace ozcode {
class Layer {
 public:
  virtual ~Layer() = default;
  Layer() = default;
  Layer(std::string const &name) : name_(name) {}
  virtual arma::mat Forward(arma::mat const &x) = 0;
  virtual arma::mat Backward(arma::mat const &x) = 0;

  std::string name() const { return name_; }

 protected:
  std::string name_;
};

}  // namespace ozcode

#endif