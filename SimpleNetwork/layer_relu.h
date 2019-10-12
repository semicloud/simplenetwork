#ifndef LAYER_RELU_H
#define LAYER_RELU_H

#include <armadillo>
#include <string>
#include "layer.h"

namespace ozcode {

class LayerRelu : public Layer {
 public:
  LayerRelu() = default;
  LayerRelu(std::string const& layer_name) : Layer(layer_name) {}

  /**
   * \brief RELU forward operation
   * \param x mat to forward
   * \return forwarded x
   */
  arma::mat Forward(arma::mat const& x) override;

  /**
   * \brief RELU backward operation
   * \param dout mat to backward
   * \return backward dout
   * \remark if the input element is greater than zero
   * \remark then the backward value is passed to the fore layer
   */
  arma::mat Backward(arma::mat const& dout) override;

 private:
  arma::umat mask_;
};
}  // namespace ozcode

#endif