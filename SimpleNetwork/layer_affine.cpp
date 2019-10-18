#include "pch.h"
#include "layer_affine.h"

ozcode::LayerAffine::LayerAffine(LayerAffine const& other) : Layer(other), w_(other.w_), b_(other.b_), x_(other.x_), dw_(other.dw_), db_(other.db_)
{
}

ozcode::LayerAffine& ozcode::LayerAffine::operator=(LayerAffine const& rhs)
{
	if (this != &rhs)
	{
		Layer::operator=(rhs);
		w_ = rhs.w_;
		b_ = rhs.b_;
		x_ = rhs.x_;
		dw_ = rhs.dw_;
		db_ = rhs.db_;
	}
	return *this;
}

ozcode::LayerAffine::LayerAffine(LayerAffine&& other) noexcept : Layer(std::move(other)), w_(other.w_), b_(other.b_), x_(other.x_), dw_(other.dw_), db_(other.db_)
{
	other.w_ = nullptr;
	other.b_ = nullptr;
}

ozcode::LayerAffine& ozcode::LayerAffine::operator=(LayerAffine&& rhs) noexcept
{
	if (this != &rhs)
	{
		Layer::operator=(rhs);
		w_ = rhs.w_;
		b_ = rhs.b_;
		x_ = rhs.x_;
		dw_ = rhs.dw_;
		db_ = rhs.db_;
		rhs.w_ = nullptr;
		rhs.b_ = nullptr;
	}
	return *this;
}

arma::mat ozcode::LayerAffine::Forward(arma::mat const& x) {
	x_ = x;
	const arma::uword batch_size = x.n_rows;
	// broadcast vector b
	const arma::mat rep_b = repmat(*b_, batch_size, 1);
	arma::arma_assert_mul_size(x, *w_, "incorrect size for multiplication");
	arma::mat out = x * (*w_) + rep_b;
	// out.print("out");
	return out;
}

arma::mat ozcode::LayerAffine::Backward(arma::mat const& dout) {
	arma::mat dx = dout * (*w_).t();

	// dW
	const std::string err_msg1 =
		"incorrect size for multiplication! \n in ozcode::layer_affine::backward "
		"1";
	const arma::mat m_x_t = x_.t();
	arma::arma_assert_mul_size(m_x_t, dout, err_msg1.c_str());
	dw_ = m_x_t * dout;

	// db, sum by column
	db_ = arma::sum(dout, 0);
	const std::string err_msg2 =
		"incorrect matrix size! \n in ozcode::layer_affine::backward 2";
	arma::arma_assert_same_size(*b_, db_, err_msg2.c_str());

	return dx;
}

ozcode::LayerAffine* ozcode::LayerAffine::clone() const
{
	return new LayerAffine(*this);
}
