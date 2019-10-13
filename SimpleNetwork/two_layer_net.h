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
			double weight_init_std = 0.01);
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
