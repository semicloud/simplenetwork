#pragma once

#include <armadillo>
#include <string>

namespace ozcode {
	class layer {
	protected:
		std::string m_name;
	public:
		virtual ~layer() = default;
		layer() = default;
		layer(std::string const& name) : m_name(name) {}
		virtual arma::mat forward(arma::mat const &x) = 0;
		virtual arma::mat backward(arma::mat const &x) = 0;

		std::string name() const { return m_name; }
	};

} // namespace ozcode
