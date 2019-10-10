#include "pch.h"
#include "gradient.h"

arma::mat ozcode::numerical_gradient(std::function<double(arma::mat const&)> const & f, arma::mat & x)
{
	const double h = 1e-4;
	arma::mat grad = arma::mat(x.n_rows, x.n_cols, arma::fill::zeros);
	std::cout << x.n_rows << ", " << x.n_cols << std::endl;
	for (arma::uword i = 0; i != x.n_rows; i++)
	{
		for (arma::uword j = 0; j != x.n_cols; j++)
		{
			const double tmp_val = x(i, j);
			x(i, j) = tmp_val + h;
			const double fxh1 = f(x);

			x(i, j) = tmp_val - h;
			const double fxh2 = f(x);

			//std::cout << "fxh1: " << fxh1 << ", fxh2: " << fxh2 << ", abs: " << std::abs(fxh1 - fxh2) << std::endl;

			grad(i, j) = (fxh1 - fxh2) / (2 * h);

			x(i, j) = tmp_val;
		}
	}
	return grad;
}


/*
 *
	const auto f = [](arma::vec const& x)
	{
		return std::pow(x(0), 2) + std::pow(x(1), 2);
	};

	std::cout << ozcode::numerical_gradient(f, arma::vec{ 3,4 }).t() << std::endl;
	std::cout << ozcode::numerical_gradient(f, arma::vec{ 0,2 }).t() << std::endl;
	std::cout << ozcode::numerical_gradient(f, arma::vec{ 3,0 }).t() << std::endl;
 *
 *
 */