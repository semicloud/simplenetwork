#include "pch.h"
#include "layer_sigmoid.h"

arma::mat ozcode::layer_sigmoid::forward(arma::mat const& x)
{
	const arma::mat out = 1.0 / (1 + arma::exp(-x));
	m_out = out;
	return out;
}

arma::mat ozcode::layer_sigmoid::backward(arma::mat const& dout)
{
	arma::arma_assert_same_size(m_out, dout,
		"layer_sigmoid backward matrix size incorrect!");
	arma::mat dx = dout % (1.0 - m_out) % m_out;
	return dx;
}

