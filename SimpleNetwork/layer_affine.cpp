#include "pch.h"
// a comment
#include "layer_affine.h"

arma::mat ozcode::LayerAffine::Forward(arma::mat const& x) {
  x_ = x;
  const arma::uword batch_size = x.n_rows;
  // broadcast vector b
  const arma::mat rep_b = repmat(*b_, batch_size, 1);
  arma::arma_assert_mul_size(x, *w_, "incorrect size for multiplication");
  arma::mat out = x * (*w_) + rep_b;
  // out.print("out");
  return out;
}

arma::mat ozcode::LayerAffine::Backward(arma::mat const& dout) {
  arma::mat dx = dout * (*w_).t();

  // dW
  const std::string err_msg1 =
      "incorrect size for multiplication! \n in ozcode::layer_affine::backward "
      "1";
  const arma::mat m_x_t = x_.t();
  arma::arma_assert_mul_size(m_x_t, dout, err_msg1.c_str());
  dw_ = m_x_t * dout;

  // db, sum by column
  db_ = arma::sum(dout, 0);
  const std::string err_msg2 =
      "incorrect matrix size! \n in ozcode::layer_affine::backward 2";
  arma::arma_assert_same_size(*b_, db_, err_msg2.c_str());

  return dx;
}
