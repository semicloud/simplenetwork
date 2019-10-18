#ifndef OPTIMIZER_ADAGRAD_H
#define OPTIMIZER_ADAGRAD_H

#include "optimizer.h"

namespace ozcode
{
	/**
	 * \brief Optimizer Adaptive SGD
	 */
	class OptimizerAdaGrad : public Optimizer {
	public:
		OptimizerAdaGrad() = default;
		OptimizerAdaGrad(double learning_rate) : Optimizer(learning_rate) {}

		OptimizerAdaGrad(const OptimizerAdaGrad& other) : Optimizer(other), h_(other.h_) {}

		OptimizerAdaGrad(OptimizerAdaGrad&& other) noexcept : Optimizer(std::move(other)), h_(std::move(other.h_)) { other.h_.clear(); }

		OptimizerAdaGrad& operator=(const OptimizerAdaGrad& other)
		{
			if (this == &other)
				return *this;
			Optimizer::operator =(other);
			h_ = other.h_;
			return *this;
		}

		OptimizerAdaGrad& operator=(OptimizerAdaGrad&& other) noexcept
		{
			if (this == &other)
				return *this;
			Optimizer::operator =(std::move(other));
			h_ = std::move(other.h_);
			return *this;
		}

		~OptimizerAdaGrad() = default;
		void Update(std::map<std::string, arma::mat>& params,
			std::map<std::string, arma::mat> const& grads) override;

	private:
		std::map<std::string, arma::mat> h_;
	};
}



#endif
