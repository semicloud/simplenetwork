#ifndef WEIGHT_DISTRIBUTION_H
#define WEIGHT_DISTRIBUTION_H

/**
 * 验证权重值的分布对神经网络的影响
 */
namespace  ozcode
{
	struct WeightInfo
	{
		int num;
		arma::mat mat;

		/**
		 * \brief 保存至文件
		 * \param dir 
		 * \return 
		 */
		int SaveFile(std::filesystem::path const& dir) const;
	};

	class WeightDistribution
	{
	public:
		WeightDistribution() = delete;
		WeightDistribution(unsigned hidden_layer_size, unsigned hidden_node_num) :
			hidden_layer_size_(hidden_layer_size), hidden_node_num_(hidden_node_num) {}
		~WeightDistribution() = default;

		/**
		 * \brief 获取每一层的权重信息
		 * \param x
		 * \return
		 */
		std::vector<WeightInfo> GetWeightInfoEveryLayer(arma::mat const& x) const;
	private:
		// 隐藏层的个数
		unsigned hidden_layer_size_;
		// 隐藏层的节点数
		unsigned hidden_node_num_;
	};
}

#endif
