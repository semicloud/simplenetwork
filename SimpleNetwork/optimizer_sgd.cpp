#include "pch.h"
// disable clang-format reorder includes
#include "assert.h"
#include "collection_utils.h"
#include "optimizer_sgd.h"

void ozcode::OptimizerSgd::Update(std::map<std::string, arma::mat>& params,
	std::map<std::string, arma::mat> const& grads)
{
	const auto pkeys = ozcode::GetKeys(params);
	const auto gkeys = ozcode::GetKeys(grads);
	assert(pkeys == gkeys);
	// std::cout << "learning rate: " << m_learning_rate << std::endl;
	for (const std::string& key : pkeys)
	{
		assert(params.find(key) != params.end());
		assert(grads.find(key) != grads.end());
		params[key] -= (learning_rate_ * (grads.find(key)->second));
	}
}
