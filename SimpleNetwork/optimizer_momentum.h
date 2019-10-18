#ifndef OPTIMIZER_MOMENTUM_H
#define OPTIMIZER_MOMENTUM_H

#include "optimizer.h"

namespace ozcode {

	/**
	 * \brief Optimizer SGD with momentum
	 */
	class OptimizerMomentum : public Optimizer
	{
	public:
		OptimizerMomentum() : Optimizer(0.01) {}
		OptimizerMomentum(double learning_rate) : Optimizer(learning_rate) {}
		OptimizerMomentum(double learning_rate, double momentum)
			: Optimizer(learning_rate), momentum_(momentum) {}


		OptimizerMomentum(const OptimizerMomentum& other) : Optimizer(other), momentum_(other.momentum_), v_(other.v_) {}

		OptimizerMomentum(OptimizerMomentum&& other) noexcept : Optimizer(std::move(other)), momentum_(other.momentum_), v_(std::move(other.v_)) {}

		OptimizerMomentum& operator=(const OptimizerMomentum& other)
		{
			if (this == &other)
				return *this;
			Optimizer::operator =(other);
			momentum_ = other.momentum_;
			v_ = other.v_;
			return *this;
		}

		OptimizerMomentum& operator=(OptimizerMomentum&& other) noexcept
		{
			if (this == &other)
				return *this;
			Optimizer::operator =(std::move(other));
			momentum_ = other.momentum_;
			v_ = std::move(other.v_);
			return *this;
		}

		void Update(std::map<std::string, arma::mat>& params,
			std::map<std::string, arma::mat> const& grads) override;

	private:
		double momentum_ = 0.9;  // 类的私有成员采用变量名后跟下划线的方式命名
		std::map<std::string, arma::mat> v_;
	};
}  // namespace ozcode

#endif
