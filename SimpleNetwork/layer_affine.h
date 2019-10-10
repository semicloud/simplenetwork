#pragma once
#include <armadillo>
#include "layer.h"

namespace ozcode
{
	class layer_affine : public layer
	{
	private:
		arma::mat* m_w;
		// as numpy, b is a row vector
		arma::mat* m_b;
		arma::mat m_x;
		arma::mat m_dw;
		arma::mat m_db;
	public:
		layer_affine() = delete;
		layer_affine(std::string const & name, arma::mat* w, arma::mat* b) : layer(name), m_w(w), m_b(b) {}

		arma::mat forward(arma::mat const & x) override;
		/**
		 * \brief Layer affine backward operation
		 * \param dout a matrix passed by the layer of softmax-loss, which is a samp_num*feature_num matrix
		 * \return
		 */
		arma::mat backward(arma::mat const & dout) override;

		/**
		 * \brief get the dW which is updated when backward
		 * \return
		 */
		arma::mat dW() const
		{
			return m_dw;
		}

		/**
		 * \brief
		 * \return get the db which is updated when backward
		 */
		arma::mat db() const
		{
			return m_db;
		}
	};
}
