#ifndef N_LAYER_NETWORK_H
#define N_LAYER_NETWORK_H

#include "layer_softmax_with_loss.h"
#include "layer.h"
#include "weight_init_mechanism.h"
#include <armadillo>

namespace ozcode
{
	/**
	 * \brief 多层神经网络
	 */
	class NLayerNetwork
	{
	public:
		NLayerNetwork();

		/**
		 * \brief 构建N层神经网络
		 * \param input_size 输入层神经元个数
		 * \param hidden_layer_node_nums 各个隐藏层神经元个数
		 * \param output_size 输出层神经元个数
		 * \remark 初始权重系数默认取值为0.01，即标准差为1的正态分布；
		 * \remark Xavier初始值适用于Sigmoid或Tanh激活函数，而He初始值适用于RELU激活函数
		 */
		NLayerNetwork(arma::uword input_size,
			std::vector<arma::uword> const& hidden_layer_node_nums,
			arma::uword output_size);

		/**
		 * \brief 构建N层神经网络
		 * \param input_size 输入层神经元个数
		 * \param hidden_layer_node_nums 各个隐藏层神经元个数
		 * \param output_size 输出层神经元个数
		 * \param mechanism 初始权重生成机制
		 */
		NLayerNetwork(arma::uword input_size,
			std::vector<arma::uword> const& hidden_layer_node_nums,
			arma::uword output_size,
			ozcode::WeightInitMechanism mechanism);

		NLayerNetwork(const NLayerNetwork& other);

		NLayerNetwork& operator=(const NLayerNetwork& rhs);

		NLayerNetwork(NLayerNetwork&& other) noexcept;

		NLayerNetwork& operator=(NLayerNetwork&& rhs) noexcept;

		~NLayerNetwork();

		/**
		 * \brief Print the information of network
		 */
		void Print(std::ostream& os) const;

		double CalculateLoss(arma::mat const& x, arma::mat const& t);

		double CalculateAccuracy(arma::mat const& x, arma::mat const& t);

		arma::mat Predict(arma::mat const& x);

		std::map<std::string, arma::mat> CalculateGradient(arma::mat const& x, arma::mat const& t);

		std::map<std::string, arma::mat>& params() { return params_; }

		void SaveWeights(std::string const& dir);

	private:
		arma::uword input_size_; // 输入层神经元个数
		std::vector<arma::uword> hidden_layer_node_nums_; // 各个隐藏层神经元的个数
		arma::uword hidden_layer_num_; //隐藏层个数
		arma::uword output_size_; // 输出层神经元个数
		ozcode::WeightInitMechanism weight_init_mechanism_ = Sigma;
		//double weight_scale_ = 0.01; // 初始权重系数
		std::map<std::string, arma::mat> params_;  // 参数
		std::vector<ozcode::Layer*> layers_;  // 隐藏层
		ozcode::LayerSoftmaxWithLoss* last_layer_;  // SoftmaxWithLoss层

		/**
		 * \brief 获取权重初始值
		 * \param size 神经元个数
		 * \return
		 */
		double GetWeightScale(arma::uword size);

		/**
		 * \brief 获取各层权重的形状（除Softmax层外）
		 * \return 各层权重的形状
		 * \remark  注意，调用该方法前，hidden_layer_node_nums_、input_size_、output_size_成员必须初始化
		 */
		std::vector<std::pair<arma::uword, arma::uword>> GetLayerShapes();

		static std::string GetName(std::string const& prefix, const arma::uword i)
		{
			return (boost::format("%1%%2%") % prefix % i).str();
		}

		void free();
	};

}

#endif
