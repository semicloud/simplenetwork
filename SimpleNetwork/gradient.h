#pragma once

#include <armadillo>

namespace ozcode
{
	/**
	 * \brief compute the gradient value using numerical approach
	 * \param f a function which return double, the parameter is a matrix
	 * \param x the input
	 * \return the gradient of f
	 */
	arma::mat numerical_gradient(std::function<double(arma::mat const&)> const& f, arma::mat & x);
}
