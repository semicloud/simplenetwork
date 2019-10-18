#ifndef N_LAYER_NETWORK_H
#define N_LAYER_NETWORK_H

#include "layer_softmax_with_loss.h"
#include "layer.h"
#include "weight_init_mechanism.h"
#include <armadillo>

namespace ozcode
{
	/**
	 * \brief ���������
	 */
	class NLayerNetwork
	{
	public:
		NLayerNetwork();

		/**
		 * \brief ����N��������
		 * \param input_size �������Ԫ����
		 * \param hidden_layer_node_nums �������ز���Ԫ����
		 * \param output_size �������Ԫ����
		 * \remark ��ʼȨ��ϵ��Ĭ��ȡֵΪ0.01������׼��Ϊ1����̬�ֲ���
		 * \remark Xavier��ʼֵ������Sigmoid��Tanh���������He��ʼֵ������RELU�����
		 */
		NLayerNetwork(arma::uword input_size,
			std::vector<arma::uword> const& hidden_layer_node_nums,
			arma::uword output_size);

		/**
		 * \brief ����N��������
		 * \param input_size �������Ԫ����
		 * \param hidden_layer_node_nums �������ز���Ԫ����
		 * \param output_size �������Ԫ����
		 * \param mechanism ��ʼȨ�����ɻ���
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
		arma::uword input_size_; // �������Ԫ����
		std::vector<arma::uword> hidden_layer_node_nums_; // �������ز���Ԫ�ĸ���
		arma::uword hidden_layer_num_; //���ز����
		arma::uword output_size_; // �������Ԫ����
		ozcode::WeightInitMechanism weight_init_mechanism_ = Sigma;
		//double weight_scale_ = 0.01; // ��ʼȨ��ϵ��
		std::map<std::string, arma::mat> params_;  // ����
		std::vector<ozcode::Layer*> layers_;  // ���ز�
		ozcode::LayerSoftmaxWithLoss* last_layer_;  // SoftmaxWithLoss��

		/**
		 * \brief ��ȡȨ�س�ʼֵ
		 * \param size ��Ԫ����
		 * \return
		 */
		double GetWeightScale(arma::uword size);

		/**
		 * \brief ��ȡ����Ȩ�ص���״����Softmax���⣩
		 * \return ����Ȩ�ص���״
		 * \remark  ע�⣬���ø÷���ǰ��hidden_layer_node_nums_��input_size_��output_size_��Ա�����ʼ��
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
