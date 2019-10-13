// SimpleNetwork.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"
// a comment
#include "simple_network.h"
#include <armadillo>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include "collection_utils.h"
#include "optimizer_sgd.h"
#include "two_layer_net.h"
#include "optimizer_momentum.h"
#include "optimizer_adagrad.h"

const unsigned kHiddenSize = 50;
const int kIterNum = 2000;
const arma::uword kBatchSize = 100;
const double kLearningRate = 0.1;

int main()
{
	arma::mat x_train, t_train, x_test, t_test;
	LoadData(data_dir(), x_train, t_train, x_test, t_test);

	const unsigned input_size = static_cast<unsigned>(x_train.n_cols);
	const unsigned output_size = static_cast<unsigned>(t_train.n_cols);

	const arma::uword train_size = x_train.n_rows;
	const arma::uword iter_per_epoch = std::max(train_size / kBatchSize, 1ull);

	std::cout.precision(8);
	std::cout.setf(std::ios::fixed);

	//! Line below is very important
	//! Armadillo will not generate random permutations without this line
	arma::arma_rng::set_seed_random();

	auto optimizer_settings = GetOptimizerSetting();

	for (auto const& setting : optimizer_settings)
	{
		std::cout << "for " << setting.first << std::endl;

		std::ofstream ofs(setting.first, std::ios::app);
		ofs.precision(8);
		ofs.setf(std::ios::fixed);

		ozcode::TwoLayerNet network(input_size, kHiddenSize, output_size);

		for (arma::uword i = 0; i != kIterNum; i++) {
			arma::uvec batch_mask = arma::randperm(train_size, kBatchSize);
			arma::mat x_batch = x_train.rows(batch_mask);
			arma::mat t_batch = t_train.rows(batch_mask);

			
			std::map<std::string, arma::mat> grad =
				network.CalculateGradient(x_batch, t_batch);

			setting.second->Update(network.params(), grad);

			double loss = network.CalculateLoss(x_train, t_train);

			std::cout << loss << std::endl;
			ofs << loss << std::endl;
		}

	}
	std::cout << "Done!" << std::endl;
}

void LoadData(std::filesystem::path const& dir, arma::mat& x_train,
	arma::mat& t_train, arma::mat& x_test, arma::mat& t_test) {
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
	std::cout << "... x_train[" << x_train.n_rows << ", " << x_train.n_cols
		<< "] loaded ..." << std::endl;
	std::cout << "... t_train[" << t_train.n_rows << ", " << t_train.n_cols
		<< "] loaded ..." << std::endl;
	std::cout << "... x_test[" << x_test.n_rows << ", " << x_test.n_cols
		<< "] loaded ..." << std::endl;
	std::cout << "... t_test[" << t_test.n_rows << ", " << t_test.n_cols
		<< "] loaded ..." << std::endl;
}

std::map<std::filesystem::path, std::shared_ptr<ozcode::Optimizer>> GetOptimizerSetting()
{
	std::map<std::filesystem::path, std::shared_ptr<ozcode::Optimizer>> settings;

	const auto output_dir = experiment_dir() / "optimize";
	std::filesystem::path sgd_output_path = output_dir / "sgd.txt";
	remove_if_exist(sgd_output_path);
	std::filesystem::path momentum_output_path = output_dir / "momentum.txt";
	remove_if_exist(momentum_output_path);
	std::filesystem::path adagrad_output_path = output_dir / "adagrad.txt";
	remove_if_exist(adagrad_output_path);

	std::shared_ptr<ozcode::Optimizer> sgd(new ozcode::OptimizerSgd(kLearningRate));
	std::shared_ptr<ozcode::Optimizer> momentum(new ozcode::OptimizerMomentum(kLearningRate));
	std::shared_ptr<ozcode::Optimizer> adagrad(new ozcode::OptimizerAdaGrad(kLearningRate));

	settings.insert(std::make_pair(sgd_output_path, sgd));
	settings.insert(std::make_pair(momentum_output_path, momentum));
	settings.insert(std::make_pair(adagrad_output_path, adagrad));

	return settings;
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
