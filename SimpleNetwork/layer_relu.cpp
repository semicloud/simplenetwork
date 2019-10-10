#include "pch.h"
#include "layer_relu.h"


arma::mat ozcode::layer_relu::forward(arma::mat const &x)
{
	m_mask = arma::find(x <= 0);
	arma::mat out = x;
	out.elem(m_mask).zeros();
	return out;
}

arma::mat ozcode::layer_relu::backward(arma::mat const& dout)
{
	arma::mat tmp = dout;
	tmp.elem(m_mask).zeros();
	return tmp;
}
