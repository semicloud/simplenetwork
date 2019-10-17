#ifndef LAYER_H
#define LAYER_H

#include <armadillo>
#include <string>

namespace ozcode {
	class Layer {
	public:
		virtual ~Layer() = default;
		Layer() = default;
		Layer(std::string const &name) : name_(name) {}
		virtual arma::mat Forward(arma::mat const &x) = 0;
		virtual arma::mat Backward(arma::mat const &x) = 0;

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