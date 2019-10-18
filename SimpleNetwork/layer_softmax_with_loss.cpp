#include "pch.h"
// a comment
#include "layer_softmax_with_loss.h"

double ozcode::LayerSoftmaxWithLoss::Forward(arma::mat const& x,
	arma::mat const& t) {
	t_ = t;
	y_ = ozcode::Softmax(x, 1);
	loss_ = ozcode::CrossEntropyLoss(y_, t_);
	return loss_;
}

arma::mat ozcode::LayerSoftmaxWithLoss::Backward(double dout = 1) {
	const arma::uword batch_size = t_.n_rows;
	// dx is a m*n matrix, which m is record number, n is feature dim
	arma::mat dx = (y_ - t_) / double(batch_size);
	return dx;
}
