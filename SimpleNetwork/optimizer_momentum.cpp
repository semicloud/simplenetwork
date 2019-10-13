#include "pch.h"
// a comment
#include "collection_utils.h"
#include "optimizer_momentum.h"

void ozcode::OptimizerMomentum::Update(
	std::map<std::string, arma::mat>& params,
	std::map<std::string, arma::mat> const& grads) {

	const std::vector<std::string> keys = ozcode::GetKeys(grads);

	for (std::string const& key : keys) {
		if (v_.find(key) == v_.end()) {
			const arma::mat tmp = grads.find(key)->second;
			v_[key] = arma::mat(arma::size(tmp), arma::fill::zeros);
		}
	}

	for (std::string const& key : keys)
	{
		arma::mat dw = grads.find(key)->second;
		v_[key] = momentum_ * v_[key] - learning_rate_ * dw;
		params[key] += v_[key];
	}
}