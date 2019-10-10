#pragma once

#include <armadillo>
#include <string>
#include "layer.h"

namespace ozcode
{
	class layer_sigmoid : public layer
	{
	private:
		arma::mat m_out;

	public:
		layer_sigmoid() = default;
		layer_sigmoid(std::string const& layer_name) : layer(layer_name) {}
		~layer_sigmoid() = default;
		arma::mat forward(arma::mat const& x) override;
		arma::mat backward(arma::mat const & dout) override;
	};
}
