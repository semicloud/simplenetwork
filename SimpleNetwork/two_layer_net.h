#pragma once

#include <armadillo>
#include <map>
#include <vector>
#include "layer.h"
#include "layer_affine.h"
#include "layer_relu.h"
#include "layer_softmax_with_loss.h"

namespace ozcode
{
	class two_layer_net
	{
	public:
		two_layer_net() = delete;
		two_layer_net(unsigned input_size, unsigned hidden_size,
			unsigned output_size, double weight_init_std = 0.01) :
			m_input_size(input_size), m_hidden_size(hidden_size),
			m_output_size(output_size), m_weight_init_std(weight_init_std)
		{
			arma::mat w1 = weight_init_std * arma::randn(m_input_size, m_hidden_size);
			// arma::mat w1;
			// w1.load("d:\\arma_w1.csv", arma::csv_ascii, true);
			// w1.save("d:\\arma_w1.csv", arma::csv_ascii, true);
			arma::mat b1 = arma::zeros(hidden_size).t(); // b1 is a row vector

			arma::mat w2 = weight_init_std * arma::randn(m_hidden_size, m_output_size);
			// w2.save("d:\\arma_w2.csv", arma::csv_ascii, true);
			// arma::mat w2;
			// w2.load("d:\\arma_w2.csv", arma::csv_ascii, true);
			arma::mat b2 = arma::zeros(output_size).t();

			m_params.insert(std::make_pair("W1", w1));
			m_params.insert(std::make_pair("b1", b1));
			m_params.insert(std::make_pair("W2", w2));
			m_params.insert(std::make_pair("b2", b2));

			m_layers.resize(3);
			//! beware that here use reference to pass the parameter
			m_layers[0] = new layer_affine("Affine1", &m_params["W1"], &m_params["b1"]);
			m_layers[1] = new layer_relu("Relu1");
			m_layers[2] = new layer_affine("Affine2", &m_params["W2"], &m_params["b2"]);

			m_softmax_with_loss = new layer_softmax_with_loss();
		}
		~two_layer_net();

		/**
		 * \brief
		 * \param x
		 * \return
		 */
		arma::mat predict(arma::mat const& x);

		/**
		 * \brief
		 * \param x
		 * \param t
		 * \return
		 */
		double loss(arma::mat const& x, arma::mat const& t);

		/**
		 * \brief
		 * \param x
		 * \param t
		 * \return
		 */
		double accuracy(arma::mat const& x, arma::mat const& t);

		/**
		 * \brief gradient descent using numerical approach
		 * \param x
		 * \param t
		 * \return
		 */
		std::map<std::string, arma::mat> numerical_gradient(arma::mat const& x, arma::mat const& t);

		/**
		 * \brief gradient descent using inverse-broadcast method
		 * \param x samples
		 * \param t labels (in one-hot manner)
		 * \return A map contains gradient (computed) matrix
		 */
		std::map<std::string, arma::mat> gradient(arma::mat const & x, arma::mat const& t);

		/**
		 * \brief get the matrix params
		 * \return std::map<std::string, arma::mat>& it is a reference object
		 */
		std::map<std::string, arma::mat>& params() { return m_params; }
	private:
		unsigned m_input_size;
		unsigned m_hidden_size;
		unsigned m_output_size;
		double m_weight_init_std;
		std::map<std::string, arma::mat> m_params;
		std::vector<ozcode::layer*> m_layers;
		layer_softmax_with_loss* m_softmax_with_loss;

		/**
		 * \brief get layers by layer_name
		 * \param layer_name layer_name
		 * \return the pointer pointed to the layer, which implements ozcode::layer
		 */
		ozcode::layer* get_layer(std::string const& layer_name);
	};


}
