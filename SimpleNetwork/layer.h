#ifndef LAYER_H
#define LAYER_H

#include <armadillo>
#include <string>

namespace ozcode {
	class Layer {
	public:
		Layer() : name_() {}
		Layer(std::string const &name) : name_(name) {}
		virtual ~Layer() {}
		Layer(Layer const& other);
		Layer& operator=(Layer const& rhs);
		Layer(Layer&& other) noexcept;
		Layer& operator=(Layer&& rhs) noexcept;

		virtual arma::mat Forward(arma::mat const &x) = 0;
		virtual arma::mat Backward(arma::mat const &x) = 0;
		virtual Layer* clone() const = 0;

		std::string name() const { return name_; }

		/**
		 * \brief Get the index of the layer, the index of a layer must be a number in layer name
		 * \return the index of layer
		 */
		int index() const;

	protected:
		std::string name_;
	};

}  // namespace ozcode

#endif