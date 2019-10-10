#include "pch.h"
#include "gradient.h"
#include "two_layer_net.h"
#include <cassert>


ozcode::two_layer_net::~two_layer_net()
{
	delete m_softmax_with_loss;
	for (layer* p_layer : m_layers)
		delete p_layer;
}

arma::mat ozcode::two_layer_net::predict(arma::mat const& x)
{
	arma::mat x_tmp(x);
	for (auto p = m_layers.begin(); p != m_layers.end(); ++p)
	{
		x_tmp = (*p)->forward(x_tmp);
	}
	return x_tmp;
}

double ozcode::two_layer_net::loss(arma::mat const& x, arma::mat const& t)
{
	const arma::mat y = predict(x);
	const double loss = m_softmax_with_loss->forward(y, t);
	return loss;
}

double ozcode::two_layer_net::accuracy(arma::mat const& x, arma::mat const& t)
{
	const arma::mat y = predict(x);
	const arma::uvec y_max_indexes = arma::index_max(y, 1);
	assert(t.n_rows > 1);
	const arma::uvec t_max_indexes = arma::index_max(t, 1);
	const double accuracy = double(accu(y_max_indexes == t_max_indexes)) / x.n_rows;
	return accuracy;
}

std::map<std::string, arma::mat>
ozcode::two_layer_net::numerical_gradient(arma::mat const& x, arma::mat const& t)
{
	const auto f = [&](arma::mat const& m) { return this->loss(x, t); };
	std::map<std::string, arma::mat> grads;
	grads["W1"] = ozcode::numerical_gradient(f, m_params["W1"]);
	grads["b1"] = ozcode::numerical_gradient(f, m_params["b1"]);
	grads["W2"] = ozcode::numerical_gradient(f, m_params["W2"]);
	grads["b2"] = ozcode::numerical_gradient(f, m_params["b2"]);
	return grads;
}

std::map<std::string, arma::mat>
ozcode::two_layer_net::gradient(arma::mat const& x, arma::mat const& t)
{
	const double loss = this->loss(x, t);
	arma::mat dout = m_softmax_with_loss->backward(1);

	for (auto it = m_layers.rbegin(); it != m_layers.rend(); ++it)
		dout = (*it)->backward(dout);

	const auto affine1 = dynamic_cast<layer_affine*>(get_layer("Affine1"));
	const auto affine2 = dynamic_cast<layer_affine*>(get_layer("Affine2"));
	std::map<std::string, arma::mat> grad;
	grad["W1"] = affine1->dW();
	grad["b1"] = affine1->db();
	grad["W2"] = affine2->dW();
	grad["b2"] = affine2->db();

	return grad;
}

ozcode::layer* ozcode::two_layer_net::get_layer(std::string const& layer_name)
{
	const auto it = std::find_if(m_layers.begin(), m_layers.end(), [&](ozcode::layer* p)
	{
		return p->name() == layer_name;
	});
	return it != m_layers.end() ? *it : nullptr;
}