#pragma once

#include <armadillo>
#include <string>
#include "layer.h"

namespace ozcode {

	class layer_relu : public layer {
	private:
		arma::umat m_mask;

	public:
		layer_relu() = default;
		layer_relu(std::string const& layer_name) : layer(layer_name) {}

		/**
		 * \brief RELU forward operation
		 * \param x mat to forward
		 * \return forwarded x
		 */
		arma::mat forward(arma::mat const &x) override;

		/**
		 * \brief RELU backward operation
		 * \param dout mat to backward
		 * \return backward dout
		 * \remark if the input element is greater than zero
		 * \remark then the backward value is passed to the fore layer
		 */
		arma::mat backward(arma::mat const& dout) override;
	};
} // namespace ozcode