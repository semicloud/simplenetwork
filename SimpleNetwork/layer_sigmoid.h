#ifndef LAYER_SIGMOID_H
#define LAYER_SIGMOID_H

#include <armadillo>
#include <string>
#include "layer.h"

namespace ozcode {
	class LayerSigmoid : public Layer
	{
	public:
		LayerSigmoid() : Layer(), out_(arma::mat(0, 0, arma::fill::zeros)) {}

		LayerSigmoid(std::string const& layer_name) : Layer(layer_name), out_(arma::mat(0, 0, arma::fill::zeros)) {}

		LayerSigmoid(const LayerSigmoid& other) : Layer(other), out_(other.out_) {}

		LayerSigmoid& operator=(const LayerSigmoid& rhs)
		{
			if (this != &rhs)
			{
				Layer::operator=(rhs);
				out_ = rhs.out_;
			}
			return *this;
		}

		LayerSigmoid(LayerSigmoid&& other) noexcept : Layer(other), out_(std::move(other.out_)) {}

		LayerSigmoid& operator=(LayerSigmoid&& rhs) noexcept
		{
			if (this != &rhs)
			{
				Layer::operator=(rhs);
				out_ = rhs.out_;
			}
			return *this;
		}

		~LayerSigmoid() = default;
		arma::mat Forward(arma::mat const& x) override;
		arma::mat Backward(arma::mat const& dout) override;
		LayerSigmoid* clone() const override;
	private:
		arma::mat out_;
	};
}  // namespace ozcode

#endif
