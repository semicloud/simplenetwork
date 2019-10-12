#ifndef LAYER_SIGMOID_H
#define LAYER_SIGMOID_H

#include <armadillo>
#include <string>
#include "layer.h"

namespace ozcode {
class LayerSigmoid : public Layer {
 public:
  LayerSigmoid() = default;
  LayerSigmoid(std::string const& layer_name) : Layer(layer_name) {}
  ~LayerSigmoid() = default;
  arma::mat Forward(arma::mat const& x) override;
  arma::mat Backward(arma::mat const& dout) override;

 private:
  arma::mat out_;
};
}  // namespace ozcode

#endif
