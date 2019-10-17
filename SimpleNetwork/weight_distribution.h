#ifndef WEIGHT_DISTRIBUTION_H
#define WEIGHT_DISTRIBUTION_H

/**
 * ��֤Ȩ��ֵ�ķֲ����������Ӱ��
 */
namespace  ozcode
{
	struct WeightInfo
	{
		int num;
		arma::mat mat;

		/**
		 * \brief �������ļ�
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
		 * \brief ��ȡÿһ���Ȩ����Ϣ
		 * \param x
		 * \return
		 */
		std::vector<WeightInfo> GetWeightInfoEveryLayer(arma::mat const& x) const;
	private:
		// ���ز�ĸ���
		unsigned hidden_layer_size_;
		// ���ز�Ľڵ���
		unsigned hidden_node_num_;
	};
}

#endif
