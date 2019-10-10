#include "pch.h"
#include "layer_affine.h"

arma::mat ozcode::layer_affine::forward(arma::mat const& x)
{
	m_x = x;
	const arma::uword batch_size = x.n_rows;
	// broadcast vector b
	const arma::mat rep_b = repmat(*m_b, batch_size, 1);
	arma::arma_assert_mul_size(x, *m_w,
		"incorrect size for multiplication");
	arma::mat out = x * (*m_w) + rep_b;
	//out.print("out");
	return out;
}

arma::mat ozcode::layer_affine::backward(arma::mat const& dout)
{
	arma::mat dx = dout * (*m_w).t();

	// dW
	const std::string err_msg1 =
		"incorrect size for multiplication! \n in ozcode::layer_affine::backward 1";
	const arma::mat m_x_t = m_x.t();
	arma::arma_assert_mul_size(m_x_t, dout, err_msg1.c_str());
	m_dw = m_x_t * dout;

	// db, sum by column
	m_db = arma::sum(dout, 0);
	const std::string err_msg2 =
		"incorrect matrix size! \n in ozcode::layer_affine::backward 2";
	arma::arma_assert_same_size(*m_b, m_db, err_msg2.c_str());

	return dx;
}
