#ifndef LAYER_AFFINE_H
#define LAYER_AFFINE_H

#include <armadillo>
#include "layer.h"

namespace ozcode {

class LayerAffine : public Layer {
 public:
  LayerAffine() = delete;
  LayerAffine(std::string const& name, arma::mat* w, arma::mat* b)
      : Layer(name), w_(w), b_(b) {}

  arma::mat Forward(arma::mat const& x) override;
  /**
   * \brief Layer affine backward operation
   * \param dout a matrix passed by the layer of softmax-loss, which is a
   * samp_num*feature_num matrix \return
   */
  arma::mat Backward(arma::mat const& dout) override;

  /**
   * \brief get the dW which is updated when backward
   * \return
   */
  arma::mat dW() const { return dw_; }

  /**
   * \brief
   * \return get the db which is updated when backward
   */
  arma::mat db() const { return db_; }

 private:
  arma::mat* w_;
  // as numpy, b is a row vector
  arma::mat* b_;
  arma::mat x_;
  arma::mat dw_;
  arma::mat db_;
};
}  // namespace ozcode

#endif
