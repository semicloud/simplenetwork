#ifndef LAYER_AFFINE_H
#define LAYER_AFFINE_H

#include "layer.h"
#include <armadillo>

namespace ozcode {

	class LayerAffine : public Layer
	{
	public:
		LayerAffine() : w_(nullptr), b_(nullptr), x_(default_mat()), dw_(default_mat()), db_(default_mat()) {}

		LayerAffine(std::string const& name, arma::mat* w, arma::mat* b)
			: Layer(name), w_(w), b_(b), x_(default_mat()), dw_(default_mat()), db_(default_mat()) {}

		LayerAffine(LayerAffine const& other);
		LayerAffine& operator=(LayerAffine const& rhs);

		LayerAffine(LayerAffine&& other) noexcept;
		LayerAffine& operator=(LayerAffine&& rhs) noexcept;

		~LayerAffine() = default;


		arma::mat Forward(arma::mat const& x) override;

		/**
		 * \brief Layer affine backward operation
		 * \param dout a matrix passed by the layer of softmax-loss, which is a
		 * samp_num*feature_num matrix \return
		 */
		arma::mat Backward(arma::mat const& dout) override;

		LayerAffine* clone() const override;

		std::pair<arma::uword, arma::uword> w_shape() const
		{
			std::pair<arma::uword, arma::uword> ans{ w_->n_rows, w_->n_cols };
			return ans;
		}

		std::pair<arma::uword, arma::uword> b_shape() const
		{
			std::pair<arma::uword, arma::uword> ans{ b_->n_rows, b_->n_cols };
			return ans;
		}

		/**
		 * \brief get the dW which is updated when backward
		 * \return
		 */
		arma::mat dW() const { return dw_; }

		/**
		 * \brief
		 * \return get the db which is updated when backward
		 */
		arma::mat db() const { return db_; }

		arma::mat W() const { return *w_; }

		/**
		 * \brief ¸üÐÂw_ºÍb_
		 * \param pw
		 * \param pb
		 */
		void UpdatePointer(arma::mat* pw, arma::mat* pb)
		{
			w_ = pw;
			b_ = pb;
		}

	private:
		arma::mat* w_;
		// as numpy, b is a row vector
		arma::mat* b_;
		arma::mat x_;
		arma::mat dw_;
		arma::mat db_;

		static arma::mat default_mat() { return arma::mat(0, 0, arma::fill::zeros); }
	};
}  // namespace ozcode

#endif
