// SimpleNetwork.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"
// a comment
#include "simple_network.h"
#include "n_layer_network.h"
#include <armadillo>
#include <filesystem>
#include <fstream>
#include <iostream>
#include "optimizer_momentum.h"
#include "optimizer_sgd.h"

const unsigned kHiddenSize = 50;
const int kIterNum = 2000;
const arma::uword kBatchSize = 128;
const double kLearningRate = 0.01;

int main()
{
	arma::mat x_train, t_train, x_test, t_test;
	LoadData(data_dir(), x_train, t_train, x_test, t_test, false);

	const arma::uword input_size = 784;
	const arma::uword node_num = 100;
	const std::vector<arma::uword> hidden_layer_node_nums{ 100,100,100,100 };
	const arma::uword output_size = 10;

	arma::arma_rng::set_seed(42);
	std::stringstream ss;
	ss.precision(9);
	ss.setf(std::ios::fixed);

	ozcode::NLayerNetwork network(input_size, hidden_layer_node_nums, output_size, ozcode::Xavier);
	network.Print(std::cout);
	ozcode::OptimizerMomentum optimizer(kLearningRate);

	for (int i = 0; i != kIterNum; ++i)
	{
		const arma::uvec choices(arma::randperm(x_train.n_rows, kBatchSize));
		const arma::mat x_batch(x_train.rows(choices));
		const arma::mat t_batch(t_train.rows(choices));
		const auto grads = network.CalculateGradient(x_batch, t_batch);
		optimizer.Update(network.params(), grads);
		double loss = network.CalculateLoss(x_batch, t_batch);
		ss << loss << "\n";
		if (i % 100 == 0)
			std::cout << "iter " << i << ", loss=" << loss << std::endl;
	}

	/*std::filesystem::path output("xavier_momentum.txt");
	if (std::filesystem::exists(output))
		std::filesystem::remove(output);
	std::ofstream ofs(output, std::ios::app);
	ofs << ss.str();
	ofs.flush();
	ofs.close();*/

	return 0;
}

void LoadData(std::filesystem::path const& dir, arma::mat& x_train,
	arma::mat& t_train, arma::mat& x_test, arma::mat& t_test, bool print) {
	if (!std::filesystem::exists(dir)) {
		std::cerr << dir << " not exist!" << std::endl;
		exit(EXIT_FAILURE);
	}

	const std::filesystem::path x_train_path = dir / "x_train.bin";
	const std::filesystem::path t_train_path = dir / "t_train.bin";
	const std::filesystem::path x_test_path = dir / "x_test.bin";
	const std::filesystem::path t_test_path = dir / "t_test.bin";
	x_train.load(x_train_path.string(), arma::arma_binary, true);
	t_train.load(t_train_path.string(), arma::arma_binary, true);
	x_test.load(x_test_path.string(), arma::arma_binary, true);
	t_test.load(t_test_path.string(), arma::arma_binary, true);
	if (print)
	{
		std::cout << "... x_train[" << x_train.n_rows << ", " << x_train.n_cols
			<< "] loaded ...\n";
		std::cout << "... t_train[" << t_train.n_rows << ", " << t_train.n_cols
			<< "] loaded ...\n";
		std::cout << "... x_test[" << x_test.n_rows << ", " << x_test.n_cols
			<< "] loaded ...\n";
		std::cout << "... t_test[" << t_test.n_rows << ", " << t_test.n_cols
			<< "] loaded ..." << std::endl;
	}
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
