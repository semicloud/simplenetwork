#include "pch.h"
#include "layer_softmax_with_loss.h"


double ozcode::layer_softmax_with_loss::forward(arma::mat const& x, arma::mat const& t)
{
	m_t = t;
	m_y = ozcode::softmax(x, 1);
	m_loss = ozcode::cross_entropy_loss(m_y, m_t);
	return m_loss;
}

arma::mat ozcode::layer_softmax_with_loss::backward(double dout = 1)
{
	const arma::uword batch_size = m_t.n_rows;
	// dx is a m*n matrix, which m is record number, n is feature dim
	arma::mat dx = (m_y - m_t) / double(batch_size);
	return dx;
}
