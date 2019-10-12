#include "pch.h"
// a comment
#include "layer_relu.h"

arma::mat ozcode::LayerRelu::Forward(arma::mat const& x) {
  mask_ = arma::find(x <= 0);
  arma::mat out = x;
  out.elem(mask_).zeros();
  return out;
}

arma::mat ozcode::LayerRelu::Backward(arma::mat const& dout) {
  arma::mat tmp = dout;
  tmp.elem(mask_).zeros();
  return tmp;
}
