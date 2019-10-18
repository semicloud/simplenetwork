#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <armadillo>
#include "my_debug.h"

namespace ozcode {

	inline arma::mat Tanh(arma::mat const& mat)
	{
		arma::mat ans = arma::tanh(mat);
		return ans;
	}

	/**
	 * \brief Softmax function
	 * \brief This function is used to transform from value to probability
	 * \param mat
	 * \return
	 */
	inline arma::mat Softmax(arma::mat const &mat) {
		const double c = mat.max();
		const arma::mat exp_a = arma::exp(mat - c);
		const double exp_sum = arma::accu(exp_a);
		const arma::mat ans = exp_a / exp_sum;
		return ans;
	}

	/**
	 * \brief Softmax to a vector
	 * \param vec a vector to softmax
	 * \return softmax value
	 */
	inline arma::vec Softmax(arma::vec const &vec) {
		const double c = vec.max();
		const arma::vec exp_a = arma::exp(vec - c);
		const double exp_sum = arma::accu(exp_a);
		const arma::vec ans = exp_a / exp_sum;
		return ans;
	}

	/**
	 * \brief Compute softmax to matrix, dim=0 as column order dim=1 as row order
	 * \param mat the matrix input
	 * \param dim 1 means compute softmax per row while 0 means per column
	 * \return softmax value
	 */
	inline arma::mat Softmax(arma::mat const &mat, int dim = 1) {
		arma::mat tmp_mat(mat.n_rows, mat.n_cols, arma::fill::zeros);
		ASSERT(dim == 0 || dim == 1, "dim only can be 0 or 1");
		const auto calculate_per_row = [&] {
			// calculating per row
			for (arma::uword i = 0; i != mat.n_rows; i++) {
				arma::vec vec = mat.row(i).t();  // making a row vector to column vector
				tmp_mat.row(i) = Softmax(vec).t();
			}
		};
		const auto calculate_per_column = [&] {
			// calculating per column
			for (arma::uword j = 0; j != mat.n_cols; j++) {
				arma::vec vec = mat.col(j);
				tmp_mat.col(j) = Softmax(vec);
			}
		};
		switch (dim) {
		case 0:
			calculate_per_column();
			break;
		case 1:
			calculate_per_row();
			break;
			// default: calculate_per_column(); break;
		}
		return tmp_mat;
	}

	/**
	 * \brief Sigmoid function
	 * \param mat
	 * \return
	 */
	inline arma::mat Sigmoid(arma::mat const &mat) {
		arma::mat ans = 1 / (1 + arma::exp(-mat));
		return ans;
	}

	/**
	 * \brief RELU function
	 * \param mat
	 * \return
	 */
	inline arma::mat Relu(arma::mat const &mat) {
		arma::mat tmp_mat(mat);
		tmp_mat.elem(arma::find(mat <= 0)).zeros();
		return tmp_mat;
	}

	/**
	 * \brief The cross entropy loss
	 * \param y the predicted value
	 * \param t the real value (usually one-hot)
	 * \return
	 */
	inline double CrossEntropyLoss(arma::mat const &y, arma::mat const &t) {
		arma::mat tmp_y = y, tmp_t = t;
		if (y.n_cols == 1 || y.n_rows == 1) {
			tmp_y.reshape(1, tmp_y.size());
			tmp_t.reshape(1, tmp_t.size());
		}
		const arma::uword batch_size = y.n_rows;
		const double delta = 1e-7;
		//! Don't forget divide total loss by batch size
		//! It's a sad story, yeah, I know.
		const double loss = -arma::accu(t % arma::log(y + delta)) / batch_size;
		return loss;
	}

}  // namespace ozcode

#endif
