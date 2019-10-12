#include "pch.h"
// a comment
#include <cassert>
#include "gradient.h"
#include "two_layer_net.h"

ozcode::TwoLayerNet::~TwoLayerNet() {
  delete layer_softmax_with_loss_;
  for (Layer* p_layer : layers_) delete p_layer;
}

arma::mat ozcode::TwoLayerNet::Predict(arma::mat const& x) {
  arma::mat x_tmp(x);
  for (auto p = layers_.begin(); p != layers_.end(); ++p) {
    x_tmp = (*p)->Forward(x_tmp);
  }
  return x_tmp;
}

double ozcode::TwoLayerNet::CalculateLoss(arma::mat const& x,
                                          arma::mat const& t) {
  const arma::mat y = Predict(x);
  const double loss = layer_softmax_with_loss_->Forward(y, t);
  return loss;
}

double ozcode::TwoLayerNet::GetAccuracy(arma::mat const& x,
                                        arma::mat const& t) {
  const arma::mat y = Predict(x);
  const arma::uvec y_max_indexes = arma::index_max(y, 1);
  assert(t.n_rows > 1);
  const arma::uvec t_max_indexes = arma::index_max(t, 1);
  const double accuracy =
      double(accu(y_max_indexes == t_max_indexes)) / x.n_rows;
  return accuracy;
}

std::map<std::string, arma::mat>
ozcode::TwoLayerNet::CalculateNumericalGradient(arma::mat const& x,
                                                arma::mat const& t) {
  const auto f = [&](arma::mat const& m) { return this->CalculateLoss(x, t); };
  std::map<std::string, arma::mat> grads;
  grads["W1"] = ozcode::CalculateNumericalGradient(f, params_["W1"]);
  grads["b1"] = ozcode::CalculateNumericalGradient(f, params_["b1"]);
  grads["W2"] = ozcode::CalculateNumericalGradient(f, params_["W2"]);
  grads["b2"] = ozcode::CalculateNumericalGradient(f, params_["b2"]);
  return grads;
}

std::map<std::string, arma::mat> ozcode::TwoLayerNet::CalculateGradient(
    arma::mat const& x, arma::mat const& t) {
  const double loss = this->CalculateLoss(x, t);
  arma::mat dout = layer_softmax_with_loss_->Backward(1);

  for (auto it = layers_.rbegin(); it != layers_.rend(); ++it)
    dout = (*it)->Backward(dout);

  const auto affine1 = dynamic_cast<LayerAffine*>(GetLayer("Affine1"));
  const auto affine2 = dynamic_cast<LayerAffine*>(GetLayer("Affine2"));
  std::map<std::string, arma::mat> grad;
  grad["W1"] = affine1->dW();
  grad["b1"] = affine1->db();
  grad["W2"] = affine2->dW();
  grad["b2"] = affine2->db();

  return grad;
}

ozcode::Layer* ozcode::TwoLayerNet::GetLayer(std::string const& layer_name) {
  const auto it =
      std::find_if(layers_.begin(), layers_.end(),
                   [&](ozcode::Layer* p) { return p->name() == layer_name; });
  return it != layers_.end() ? *it : nullptr;
}