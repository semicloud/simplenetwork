#ifndef TWO_LAYER_NET_H
#define TWO_LAYER_NET_H

#include "layer.h"
#include "layer_affine.h"
#include "layer_relu.h"
#include "layer_softmax_with_loss.h"
// a comment
#include <armadillo>
#include <map>
#include <vector>

namespace ozcode {

class TwoLayerNet {
 public:
  TwoLayerNet() = delete;
  TwoLayerNet(unsigned input_size, unsigned hidden_size, unsigned output_size,
              double weight_init_std = 0.01)
      : input_size_(input_size),
        hidden_size_(hidden_size),
        output_size_(output_size),
        weight_init_std_(weight_init_std) {
    arma::mat w1 = weight_init_std * arma::randn(input_size_, hidden_size_);
    // arma::mat w1;
    // w1.load("d:\\arma_w1.csv", arma::csv_ascii, true);
    // w1.save("d:\\arma_w1.csv", arma::csv_ascii, true);
    arma::mat b1 = arma::zeros(hidden_size).t();  // b1 is a row vector

    arma::mat w2 = weight_init_std * arma::randn(hidden_size_, output_size_);
    // w2.save("d:\\arma_w2.csv", arma::csv_ascii, true);
    // arma::mat w2;
    // w2.load("d:\\arma_w2.csv", arma::csv_ascii, true);
    arma::mat b2 = arma::zeros(output_size).t();

    params_.insert(std::make_pair("W1", w1));
    params_.insert(std::make_pair("b1", b1));
    params_.insert(std::make_pair("W2", w2));
    params_.insert(std::make_pair("b2", b2));

    layers_.resize(3);
    //! beware that here use reference to pass the parameter
    layers_[0] = new LayerAffine("Affine1", &params_["W1"], &params_["b1"]);
    layers_[1] = new LayerRelu("Relu1");
    layers_[2] = new LayerAffine("Affine2", &params_["W2"], &params_["b2"]);

    layer_softmax_with_loss_ = new LayerSoftmaxWithLoss();
  }
  ~TwoLayerNet();

  /**
   * \brief
   * \param x
   * \return
   */
  arma::mat Predict(arma::mat const& x);

  /**
   * \brief
   * \param x
   * \param t
   * \return
   */
  double CalculateLoss(arma::mat const& x, arma::mat const& t);

  /**
   * \brief
   * \param x
   * \param t
   * \return
   */
  double GetAccuracy(arma::mat const& x, arma::mat const& t);

  /**
   * \brief gradient descent using numerical approach
   * \param x
   * \param t
   * \return
   */
  std::map<std::string, arma::mat> CalculateNumericalGradient(
      arma::mat const& x, arma::mat const& t);

  /**
   * \brief gradient descent using inverse-broadcast method
   * \param x samples
   * \param t labels (in one-hot manner)
   * \return A map contains gradient (computed) matrix
   */
  std::map<std::string, arma::mat> CalculateGradient(arma::mat const& x,
                                                    arma::mat const& t);

  /**
   * \brief get the matrix params
   * \return std::map<std::string, arma::mat>& it is a reference object
   */
  std::map<std::string, arma::mat>& params() { return params_; }

 private:
  unsigned input_size_;
  unsigned hidden_size_;
  unsigned output_size_;
  double weight_init_std_;
  std::map<std::string, arma::mat> params_;
  std::vector<ozcode::Layer*> layers_;
  LayerSoftmaxWithLoss* layer_softmax_with_loss_;

  /**
   * \brief get layers by layer_name
   * \param layer_name layer_name
   * \return the pointer pointed to the layer, which implements ozcode::layer
   */
  ozcode::Layer* GetLayer(std::string const& layer_name);
};

}  // namespace ozcode

#endif
