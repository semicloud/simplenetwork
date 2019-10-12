#ifndef LAYER_SOFTMAX_WITH_LOSS_H
#define LAYER_SOFTMAX_WITH_LOSS_H

#include <armadillo>
#include "functions.h"

namespace ozcode {
class LayerSoftmaxWithLoss {
 public:
  LayerSoftmaxWithLoss() = default;
  ~LayerSoftmaxWithLoss() = default;
  /**
   * \brief forward operation
   * \param x samples
   * \param t label to samples, m * n, which m is sample num, one-hot
   * \return
   */
  double Forward(arma::mat const& x, arma::mat const& t);
  arma::mat Backward(double dout);

 private:
  double loss_ = 0;
  arma::mat t_;
  arma::mat y_;
};
}  // namespace ozcode

#endif
