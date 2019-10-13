#include "pch.h"
// a comment
#include "optimizer_adagrad.h"
#include "collection_utils.h"

void ozcode::OptimizerAdaGrad::Update(std::map<std::string, arma::mat>& params,
	std::map<std::string, arma::mat> const& grads)
{
	const std::vector<std::string> keys = ozcode::GetKeys(grads);
	for (std::string const& key : keys)
	{
		if (h_.find(key) == h_.end())
		{
			const arma::mat tmp = grads.find(key)->second;
			h_[key] = arma::mat(arma::size(tmp), arma::fill::zeros);
		}
	}

	for (std::string const& key : keys)
	{
		const arma::mat dw = grads.find(key)->second;
		h_[key] += arma::square(dw);
		params[key] -= learning_rate_ * dw / (arma::sqrt(h_[key]) + 1e-7);
	}
}
