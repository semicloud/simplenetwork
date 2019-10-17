#ifndef WEIGHT_INIT_MECHANISM_H
#define WEIGHT_INIT_MECHANISM_H

namespace ozcode
{
	/**
	 * \brief Ȩ�س�ʼ������
	 */
	enum  WeightInitMechanism
	{
		Sigma = 0, // Sigma 0.01
		Xavier = 1, // ! Xavier initialization value
		He = 2
	};
}

#endif

