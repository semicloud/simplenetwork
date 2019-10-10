#pragma once
#include <map>
#include <string>
#include <armadillo>

namespace ozcode
{
	class optimizer
	{
	public:
		optimizer() : m_learning_rate(0.01) {}
		optimizer(double lr) : m_learning_rate(lr) {}
		virtual ~optimizer() = default;

		/**
		 * \brief Update params using grads
		 * \param params
		 * \param grads
		 */
		virtual void update(std::map<std::string, arma::mat>& params,
			std::map<std::string, arma::mat> const & grads) = 0;
	protected:
		double m_learning_rate;
	};
}



