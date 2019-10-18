#include "pch.h"
// a comment
#include "layer_sigmoid.h"

arma::mat ozcode::LayerSigmoid::Forward(arma::mat const& x) {
	const arma::mat out = 1.0 / (1 + arma::exp(-x));
	out_ = out;
	return out;
}

arma::mat ozcode::LayerSigmoid::Backward(arma::mat const& dout) {
	arma::arma_assert_same_size(out_, dout,
		"layer_sigmoid backward matrix size incorrect!");
	arma::mat dx = dout % (1.0 - out_) % out_;
	return dx;
}

ozcode::LayerSigmoid* ozcode::LayerSigmoid::clone() const
{
	return new LayerSigmoid(*this);
}
