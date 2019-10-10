#pragma once
#include "optimizer.h"

namespace ozcode
{
	class optimizer_sgd : public optimizer
	{
	public:
		optimizer_sgd() : optimizer() {}
		optimizer_sgd(double learning_rate) : optimizer(learning_rate) {}
		~optimizer_sgd() = default;
		void update(std::map<std::string, arma::mat>& params, std::map<std::string, arma::mat> const& grads) override;
	};
}



