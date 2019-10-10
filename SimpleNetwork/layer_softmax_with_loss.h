#pragma once

#include <armadillo>
#include "functions.h"

namespace ozcode
{
	class layer_softmax_with_loss
	{
	public:
		layer_softmax_with_loss() = default;
		~layer_softmax_with_loss() = default;
		/**
		 * \brief forward operation
		 * \param x samples
		 * \param t label to samples, m * n, which m is sample num, one-hot
		 * \return 
		 */
		double forward(arma::mat const& x, arma::mat const& t);
		arma::mat backward(double dout);
	private:
		double m_loss = 0;
		arma::mat m_t;
		arma::mat m_y;
	};
}

