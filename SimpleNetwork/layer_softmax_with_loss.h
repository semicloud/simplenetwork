#ifndef LAYER_SOFTMAX_WITH_LOSS_H
#define LAYER_SOFTMAX_WITH_LOSS_H

#include <armadillo>
#include "functions.h"

namespace ozcode {
	class LayerSoftmaxWithLoss {
	public:
		LayerSoftmaxWithLoss() : loss_(0.0), t_(arma::mat(0, 0, arma::fill::zeros)), y_(arma::mat(0, 0, arma::fill::zeros)) {};
		~LayerSoftmaxWithLoss() {}

		LayerSoftmaxWithLoss(const LayerSoftmaxWithLoss& other) : loss_(other.loss_), t_(other.t_), y_(other.y_) {}

		LayerSoftmaxWithLoss& operator=(const LayerSoftmaxWithLoss& rhs)
		{
			if (this != &rhs)
			{
				loss_ = rhs.loss_;
				t_ = rhs.t_;
				y_ = rhs.y_;
			}
			return *this;
		}

		LayerSoftmaxWithLoss(LayerSoftmaxWithLoss&& other) noexcept : loss_(other.loss_), t_(other.t_), y_(other.y_)
		{
			other.loss_ = 0.0;
			other.t_.clear();
			other.y_.clear();
		}

		LayerSoftmaxWithLoss& operator=(LayerSoftmaxWithLoss&& rhs) noexcept
		{
			if (this != &rhs)
			{
				loss_ = rhs.loss_;
				t_ = rhs.t_;
				y_ = rhs.y_;
				rhs.loss_ = 0.0;
				rhs.t_.clear();
				rhs.y_.clear();
			}
			return *this;
		}

		/**
		 * \brief forward operation
		 * \param x samples
		 * \param t label to samples, m * n, which m is sample num, one-hot
		 * \return
		 */
		double Forward(arma::mat const& x, arma::mat const& t);
		arma::mat Backward(double dout);

	private:
		double loss_ = 0;
		arma::mat t_;
		arma::mat y_;
	};
}  // namespace ozcode

#endif
