#ifndef OPTIMIZER_SGD_H
#define OPTIMIZER_SGD_H

#include "optimizer.h"

namespace ozcode {
	class OptimizerSgd : public Optimizer {
	public:
		OptimizerSgd() {}
		OptimizerSgd(double learning_rate) : Optimizer(learning_rate) {}
		~OptimizerSgd() = default;
		void Update(std::map<std::string, arma::mat>& params,
			std::map<std::string, arma::mat> const& grads) override;
	};
}  // namespace ozcode

#endif