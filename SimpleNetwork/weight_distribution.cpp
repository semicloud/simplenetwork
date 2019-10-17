#include "pch.h"
#include "functions.h"
#include "weight_distribution.h"
#include <boost/format.hpp>
#include <cassert>


int ozcode::WeightInfo::SaveFile(std::filesystem::path const& dir) const
{
	const std::filesystem::path output_path = dir / "weight" / (boost::format("layer_%1%.txt") % num).str();
	if (!std::filesystem::exists(output_path.parent_path()))
		std::filesystem::create_directories(output_path);
	if (std::filesystem::exists(output_path))
		std::filesystem::remove(output_path);
	const arma::rowvec rvec = arma::vectorise(mat, 1);
	const arma::colvec cvec = rvec.t(); // column vector is suitable for save 
	const bool b = cvec.save(output_path.string(), arma::raw_ascii, true);
	return b ? 0 : -1;
}

std::vector<ozcode::WeightInfo> ozcode::WeightDistribution::GetWeightInfoEveryLayer(arma::mat const& x) const
{
	assert(x.n_cols == hidden_node_num_);
	arma::mat tmp_x(x);
	std::vector<WeightInfo> ans;
	ans.reserve(hidden_layer_size_);
	for (unsigned i = 0; i != hidden_layer_size_; ++i)
	{
		if (i != 0)
		{
			tmp_x = ans[i - 1].mat;
		}
		// 标准正态分布
		//arma::mat w = arma::mat(hidden_node_num_, hidden_node_num_, arma::fill::randn) * 1.0;
		// 标准差为0.01的高斯分布
		//arma::mat w = arma::mat(hidden_node_num_, hidden_node_num_, arma::fill::randn) * 0.01;
		// ! Xavier初始值
		//arma::mat w = arma::mat(hidden_node_num_, hidden_node_num_, arma::fill::randn) / (sqrt(hidden_node_num_));
		// 专门应用于RELU激活函数的He初始值
		arma::mat w = arma::mat(hidden_node_num_, hidden_node_num_, arma::fill::randn) * sqrt(2.0 / hidden_node_num_);
		arma::mat a = tmp_x * w;
		arma::mat z = ozcode::Relu(a);
		WeightInfo weight_info;
		weight_info.num = i + 1;
		weight_info.mat = z;
		ans.push_back(weight_info);
	}

	return ans;
}
