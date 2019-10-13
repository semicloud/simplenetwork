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

		~OptimizerAdaGrad() = default;
		void Update(std::map<std::string, arma::mat>& params,
			std::map<std::string, arma::mat> const& grads) override;

	private:
		std::map<std::string, arma::mat> h_;
	};
}



#endif
