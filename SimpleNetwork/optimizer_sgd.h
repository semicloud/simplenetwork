#ifndef OPTIMIZER_SGD_H
#define OPTIMIZER_SGD_H

#include "optimizer.h"

namespace ozcode {
	class OptimizerSgd : public Optimizer {
	public:
		OptimizerSgd() : Optimizer() {}
		OptimizerSgd(double learning_rate) : Optimizer(learning_rate) {}

		OptimizerSgd(const OptimizerSgd& other) : Optimizer(other)
		{
		}

		OptimizerSgd(OptimizerSgd&& other) noexcept
			: Optimizer(std::move(other))
		{
		}

		OptimizerSgd& operator=(const OptimizerSgd& other)
		{
			if (this == &other)
				return *this;
			Optimizer::operator =(other);
			return *this;
		}

		OptimizerSgd& operator=(OptimizerSgd&& other) noexcept
		{
			if (this == &other)
				return *this;
			Optimizer::operator =(std::move(other));
			return *this;
		}

		~OptimizerSgd() = default;
		void Update(std::map<std::string, arma::mat>& params,
			std::map<std::string, arma::mat> const& grads) override;
	};
}  // namespace ozcode

#endif