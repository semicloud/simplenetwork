#ifndef LAYER_RELU_H
#define LAYER_RELU_H

#include <armadillo>
#include <string>
#include "layer.h"

namespace ozcode {

	class LayerRelu : public Layer {
	public:
		LayerRelu() : Layer(""), mask_(arma::umat(0, 0, arma::fill::zeros)) {}
		LayerRelu(std::string const& layer_name) : Layer(layer_name), mask_(arma::umat(0, 0, arma::fill::zeros)) {}

		LayerRelu(const LayerRelu& other) : Layer(other), mask_(other.mask_) {}

		LayerRelu& operator=(const LayerRelu& rhs)
		{
			if (this != &rhs)
			{
				//! Remember
				Layer::operator=(rhs);
				mask_ = rhs.mask_;
			}
			return *this;
		}

		LayerRelu(LayerRelu&& other) noexcept : Layer(std::move(other)), mask_(std::move(other.mask_)) {}

		LayerRelu& operator=(LayerRelu&& rhs) noexcept
		{
			if (this != &rhs)
			{
				Layer::operator=(rhs);
				mask_ = rhs.mask_;
				rhs.mask_.clear();
			}
			return *this;
		}

		~LayerRelu() = default;

		/**
		 * \brief RELU forward operation
		 * \param x mat to forward
		 * \return forwarded x
		 */
		arma::mat Forward(arma::mat const& x) override;

		/**
		 * \brief RELU backward operation
		 * \param dout mat to backward
		 * \return backward dout
		 * \remark if the input element is greater than zero
		 * \remark then the backward value is passed to the fore layer
		 */
		arma::mat Backward(arma::mat const& dout) override;

		LayerRelu* clone() const override;

	private:
		arma::umat mask_;
	};
}  // namespace ozcode

#endif