// SimpleNetwork.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"
#include <armadillo>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <fstream>
#include "two_layer_net.h"
#include "collection_utils.h"
#include "optimizer_sgd.h"

/**
 * \brief
 * \param dir
 * \param x_train
 * \param t_train
 * \param x_test
 * \param t_test
 */
void load_data(std::filesystem::path const& dir, arma::mat& x_train
	, arma::mat & t_train, arma::mat & x_test, arma::mat& t_test);

std::filesystem::path data_dir();

int main()
{
	arma::mat x_train, t_train, x_test, t_test;
	load_data(data_dir(), x_train, t_train, x_test, t_test);

	const unsigned input_size = unsigned(x_train.n_cols);
	const unsigned hidden_size = 50;
	const unsigned output_size = unsigned(t_train.n_cols);
	ozcode::two_layer_net network(input_size, hidden_size, output_size);

	const int iter_num = 10000;
	const arma::uword train_size = x_train.n_rows;
	const arma::uword batch_size = 100;
	const double learning_rate = 0.1;

	std::vector<double> vec_train_loss;
	std::vector<double> vec_train_acc;
	std::vector<double> vec_test_acc;

	const arma::uword iter_per_epoch = std::max(train_size / batch_size, 1ull);

	std::cout.precision(8);
	std::cout.setf(std::ios::fixed);

	if (std::filesystem::exists("d:\\network_log.txt"))
		std::filesystem::remove("d:\\network_log.txt");
	std::ofstream out_file("d:\\network_log.txt", std::ios::app);
	out_file.precision(8);
	out_file.setf(std::ios::fixed);

	ozcode::optimizer*  optimizer = new ozcode::optimizer_sgd(learning_rate);

	for (arma::uword i = 0; i != iter_num; i++)
	{
		arma::uvec batch_mask = arma::randperm(train_size, batch_size);
		arma::mat x_batch = x_train.rows(batch_mask);
		arma::mat t_batch = t_train.rows(batch_mask);

		std::map<std::string, arma::mat> grad = network.gradient(x_batch, t_batch);

		optimizer->update(network.params(), grad);

		//network.params()["W1"] -= grad["W1"] * learning_rate;
		//network.params()["b1"] -= grad["b1"] * learning_rate;
		//network.params()["W2"] -= grad["W2"] * learning_rate;
		//network.params()["b2"] -= grad["b2"] * learning_rate;

		double loss = network.loss(x_train, t_train);
		vec_train_loss.push_back(loss);

		if (i % iter_per_epoch == 0)
		{
			double train_acc = network.accuracy(x_train, t_train);
			double test_acc = network.accuracy(x_test, t_test);
			vec_train_acc.push_back(train_acc);
			vec_test_acc.push_back(test_acc);
			std::cout << "train acc, test_acc: " << train_acc << ", " << test_acc << std::endl;
			out_file << "train acc, test_acc: " << train_acc << ", " << test_acc << std::endl;
		}
		std::cout << "iter " << i + 1 << ", loss: " << loss << std::endl;
		out_file << "iter " << i + 1 << ", loss: " << loss << std::endl;
	}

	delete optimizer;

	out_file.close();

	arma::vec v1(vec_train_loss);
	arma::vec v2(vec_train_acc);
	arma::vec v3(vec_test_acc);

	v1.save("d:\\train_loss.txt", arma::raw_ascii, true);
	v2.save("d:\\train_acc.txt", arma::raw_ascii, true);
	v3.save("d:\\test_acc.txt", arma::raw_ascii, true);

	std::cout << "Done!" << std::endl;
}


void load_data(std::filesystem::path const& dir,
	arma::mat& x_train, arma::mat& t_train, arma::mat& x_test, arma::mat& t_test)
{
	if (!std::filesystem::exists(dir))
	{
		std::cerr << dir << " not exist!" << std::endl;
		exit(EXIT_FAILURE);
	}

	const std::filesystem::path x_train_path
		= dir / "x_train.bin";
	const std::filesystem::path t_train_path
		= dir / "t_train.bin";
	const std::filesystem::path x_test_path =
		dir / "x_test.bin";
	const std::filesystem::path t_test_path =
		dir / "t_test.bin";
	x_train.load(x_train_path.string(), arma::arma_binary, true);
	t_train.load(t_train_path.string(), arma::arma_binary, true);
	x_test.load(x_test_path.string(), arma::arma_binary, true);
	t_test.load(t_test_path.string(), arma::arma_binary, true);
	std::cout << "... x_train[" << x_train.n_rows << ", " << x_train.n_cols << "] loaded ..." << std::endl;
	std::cout << "... t_train[" << t_train.n_rows << ", " << t_train.n_cols << "] loaded ..." << std::endl;
	std::cout << "... x_test[" << x_test.n_rows << ", " << x_test.n_cols << "] loaded ..." << std::endl;
	std::cout << "... t_test[" << t_test.n_rows << ", " << t_test.n_cols << "] loaded ..." << std::endl;
}

std::filesystem::path data_dir()
{
	return std::filesystem::current_path().parent_path() / "data" / "bin";
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门提示:
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5.
//   转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
