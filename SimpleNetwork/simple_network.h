#ifndef SIMPLE_NETWORK_H
#define SIMPLE_NETWORK_H
#include "optimizer.h"

/**
 * \brief
 * \param dir
 * \param x_train
 * \param t_train
 * \param x_test
 * \param t_test
 */
void LoadData(std::filesystem::path const& dir, arma::mat& x_train,
	arma::mat& t_train, arma::mat& x_test, arma::mat& t_test);

inline std::filesystem::path data_dir()
{
	const std::filesystem::path ans =
		std::filesystem::current_path().parent_path() / "data" / "bin";
	return ans;
}

inline std::filesystem::path experiment_dir()
{
	const std::filesystem::path ans =
		std::filesystem::current_path() / "experiment";
	if (!std::filesystem::exists(ans))
		std::filesystem::create_directories(ans);
	return ans;
}

inline void remove_if_exist(std::filesystem::path const& f)
{
	if (std::filesystem::exists(f))
		std::filesystem::remove(f);
}

/**
 * \brief 为了比较验证三种梯度下降算法的性能
 * \brief 将三种算法的输出文件和指针放到一个map中
 * \return
 */
std::map<std::filesystem::path, std::shared_ptr<ozcode::Optimizer>> GetOptimizerSetting();
#endif

