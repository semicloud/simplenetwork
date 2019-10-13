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
		void Update(std::map<std::string, arma::mat>& params,
			std::map<std::string, arma::mat> const& grads) override;

	private:
		double momentum_ = 0.9;  // ���˽�г�Ա���ñ���������»��ߵķ�ʽ����
		std::map<std::string, arma::mat> v_;
	};
}  // namespace ozcode

#endif
