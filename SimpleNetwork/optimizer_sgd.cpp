#include "pch.h"
#include "collection_utils.h"
#include "optimizer_sgd.h"
#include "assert.h"


void ozcode::optimizer_sgd::update(std::map<std::string, arma::mat>& params,
	std::map<std::string, arma::mat> const& grads)
{
	const auto pkeys = ozcode::keys(params);
	const auto gkeys = ozcode::keys(grads);
	assert(pkeys == gkeys);
	//std::cout << "learning rate: " << m_learning_rate << std::endl;
	for (const std::string& key : pkeys)
	{
		assert(params.find(key) != params.end());
		assert(grads.find(key) != grads.end());
		params[key] -= (m_learning_rate * (grads.find(key)->second));
	}
}
