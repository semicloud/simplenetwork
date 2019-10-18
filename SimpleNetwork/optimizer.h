#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <armadillo>
#include <map>
#include <string>

namespace ozcode {
	class Optimizer {
	public:
		Optimizer() : learning_rate_(0.01) {}
		Optimizer(double lr) : learning_rate_(lr) {}

		Optimizer(const Optimizer& other) : learning_rate_(other.learning_rate_) {}

		Optimizer(Optimizer&& other) noexcept
			: learning_rate_(other.learning_rate_)
		{
			other.learning_rate_ = 0.0;
		}

		Optimizer& operator=(const Optimizer& rhs)
		{
			if (this == &rhs)
				return *this;
			learning_rate_ = rhs.learning_rate_;
			return *this;
		}

		Optimizer& operator=(Optimizer&& other) noexcept
		{
			if (this == &other)
				return *this;
			learning_rate_ = other.learning_rate_;
			other.learning_rate_ = 0.0;
			return *this;
		}

		virtual ~Optimizer() = default;

		/**
		 * \brief Update params using grads
		 * \param params
		 * \param grads
		 */
		virtual void Update(std::map<std::string, arma::mat>& params,
			std::map<std::string, arma::mat> const& grads) = 0;

		//virtual Optimizer* clone();

	protected:
		double learning_rate_;
	};
}  // namespace ozcode

#endif
