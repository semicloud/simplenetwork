#include "pch.h"
#include "n_layer_network.h"
#include "layer_affine.h"
#include "layer_relu.h"
#include <deque>
#include <boost/format.hpp>


ozcode::NLayerNetwork::NLayerNetwork() : input_size_(0), hidden_layer_node_nums_(), hidden_layer_num_(0), output_size_(0), weight_init_mechanism_(Sigma), params_(), layers_(), last_layer_(nullptr)
{
}

ozcode::NLayerNetwork::NLayerNetwork(arma::uword input_size,
	std::vector<arma::uword> const& hidden_layer_node_nums,
	arma::uword output_size) :
	NLayerNetwork(input_size, hidden_layer_node_nums, output_size, Sigma)
{
}

ozcode::NLayerNetwork::NLayerNetwork(arma::uword input_size,
	std::vector<arma::uword> const& hidden_layer_node_nums,
	arma::uword output_size,
	ozcode::WeightInitMechanism weight_init_mechanism) :
	input_size_(input_size),
	hidden_layer_node_nums_(hidden_layer_node_nums),
	hidden_layer_num_(hidden_layer_node_nums.size()),
	output_size_(output_size),
	weight_init_mechanism_(weight_init_mechanism),
	params_(), layers_(), last_layer_(nullptr)
{
	//! 调用GetLayerShapes前必须初始化hidden_layer_node_nums_、input_size_、output_size_成员
	std::vector<std::pair<arma::uword, arma::uword>> const layer_shapes = GetLayerShapes();
	for (size_t i = 0; i != layer_shapes.size(); ++i)
	{
		const std::pair<arma::uword, arma::uword> layer_shape = layer_shapes.at(i);
		const arma::uword row = layer_shape.first, col = layer_shape.second;

		const double weight_scale = GetWeightScale(row); // row 也是?前一层?神经元的个数

		arma::mat w(row, col, arma::fill::randn);
		w *= weight_scale;
		arma::mat b(1, col, arma::fill::zeros);
		const std::string w_name = GetName("W", i), b_name = GetName("b", i);
		params_[w_name] = w;
		params_[b_name] = b;
		//! 这里必须取params_中w和b的地址
		Layer* p_layer_affine = new LayerAffine(GetName("Affine", i), &params_[w_name], &params_[b_name]);
		layers_.push_back(p_layer_affine);
		if (i != layer_shapes.size() - 1)
		{
			Layer* p_layer_relu = new LayerRelu(GetName("Relu", i));
			layers_.push_back(p_layer_relu);
		}
	}

	last_layer_ = new ozcode::LayerSoftmaxWithLoss();
}

ozcode::NLayerNetwork::NLayerNetwork(const NLayerNetwork& other)
	: input_size_(other.input_size_), hidden_layer_node_nums_(other.hidden_layer_node_nums_),
	hidden_layer_num_(other.hidden_layer_num_), output_size_(other.output_size_),
	weight_init_mechanism_(other.weight_init_mechanism_), params_(other.params_), layers_(), last_layer_()
{
	/* 这里，在拷贝了params_之后，params_中参数矩阵的地址已经改了，因此这里需要更新一下AffineLayer中的指针地址*/
	/* ! 获取有更好的设计？是否应该禁止拷贝这个对象？那应该如何把该对象放入vector中呢？ */
	for (Layer* layer : other.layers_)
	{
		Layer* p_layer = layer->clone();
		LayerAffine* p_layer_affine = dynamic_cast<LayerAffine*>(p_layer);
		if (p_layer_affine != nullptr)
		{
			const std::string w_name = GetName("W", p_layer_affine->index()), b_name = GetName("b", p_layer_affine->index());
			p_layer_affine->UpdatePointer(&params_[w_name], &params_[b_name]);
		}
		layers_.push_back(p_layer);
	}
	last_layer_ = new LayerSoftmaxWithLoss(*other.last_layer_);
}

ozcode::NLayerNetwork& ozcode::NLayerNetwork::operator=(const NLayerNetwork& rhs)
{
	if (this != &rhs)
	{
		free();
		input_size_ = rhs.input_size_;
		hidden_layer_node_nums_ = rhs.hidden_layer_node_nums_;
		hidden_layer_num_ = rhs.hidden_layer_num_;
		output_size_ = rhs.output_size_;
		weight_init_mechanism_ = rhs.weight_init_mechanism_;
		params_ = rhs.params_;
		for (Layer* layer : rhs.layers_)
			layers_.push_back(layer->clone());
		last_layer_ = new LayerSoftmaxWithLoss(*rhs.last_layer_);
	}
	return *this;
}

ozcode::NLayerNetwork::NLayerNetwork(NLayerNetwork&& other) noexcept
	: input_size_(other.input_size_), hidden_layer_node_nums_(std::move(other.hidden_layer_node_nums_)),
	hidden_layer_num_(other.hidden_layer_num_), output_size_(other.output_size_),
	weight_init_mechanism_(other.weight_init_mechanism_), params_(std::move(other.params_)),
	layers_(), last_layer_(nullptr)
{
	for (Layer* layer : other.layers_)
		layers_.push_back(layer->clone());
	last_layer_ = new LayerSoftmaxWithLoss(*other.last_layer_);
	other.free();
}

ozcode::NLayerNetwork& ozcode::NLayerNetwork::operator=(NLayerNetwork&& rhs) noexcept
{
	if (this != &rhs)
	{
		free();
		input_size_ = rhs.input_size_;
		hidden_layer_node_nums_ = rhs.hidden_layer_node_nums_;
		hidden_layer_num_ = rhs.hidden_layer_num_;
		output_size_ = rhs.output_size_;
		weight_init_mechanism_ = rhs.weight_init_mechanism_;
		params_ = rhs.params_;
		for (Layer* layer : rhs.layers_)
			layers_.push_back(layer->clone());
		last_layer_ = new LayerSoftmaxWithLoss(*rhs.last_layer_);
		rhs.input_size_ = 0;
		rhs.hidden_layer_node_nums_.clear();
		rhs.hidden_layer_num_ = 0;
		rhs.output_size_ = 0;
		rhs.params_.clear();
		rhs.free();
	}
	return *this;
}


ozcode::NLayerNetwork::~NLayerNetwork()
{
	delete last_layer_;
	last_layer_ = nullptr;
	for (Layer* hidden_layer : layers_)
		delete hidden_layer;
	layers_.clear();
}



void ozcode::NLayerNetwork::Print(std::ostream& os) const
{
	os << hidden_layer_num_ << " layers in this network.\n";
	for (Layer* layer : layers_)
	{
		os << layer->name();
		const LayerAffine* p = dynamic_cast<LayerAffine*>(layer);
		if (p != nullptr)
		{
			os << ",Affine, shape: " << p->w_shape().first << "x" << p->w_shape().second
				<< "sum=" << arma::accu(p->W()) << ", b shape: " << p->b_shape().first << "x" << p->b_shape().second << "\n";
			os << "mean: " << arma::accu(p->W()) / p->W().size() << "\n";
		}
		else
		{
			os << ",Relu\n";
		}
	}
}

double ozcode::NLayerNetwork::CalculateLoss(arma::mat const& x, arma::mat const& t)
{
	//std::cout << arma::accu(dynamic_cast<LayerAffine*>(layers_[0])->W()) << std::endl;
	const arma::mat y_hat = Predict(x);
	const double loss = last_layer_->Forward(y_hat, t);
	return loss;
}

double ozcode::NLayerNetwork::CalculateAccuracy(arma::mat const& x, arma::mat const& t)
{
	const arma::mat y_hat = Predict(x);
	const arma::uvec y_hat_max_indexes = arma::index_max(y_hat, 1); // in row manner
	assert(t.n_rows > 1);
	const arma::uvec t_max_indexes = arma::index_max(t, 1);
	const double accuracy =
		static_cast<double>(arma::accu(y_hat_max_indexes == t_max_indexes)) / x.n_rows;
	return accuracy;
}

arma::mat ozcode::NLayerNetwork::Predict(arma::mat const& x)
{
	arma::mat x_tmp(x);
	for (std::vector<ozcode::Layer*>::const_iterator it = layers_.begin();
		it != layers_.end(); ++it)
	{
		x_tmp = (*it)->Forward(x_tmp);
	}
	//! Wrong if use softmax
	//! 如果在这里使用Softmax函数的话，相当于Softmax的值算了两次，horrible~
	//! arma::mat ans = ozcode::Softmax(x_tmp, 1);
	arma::mat ans = x_tmp;
	return ans;
}

std::map<std::string, arma::mat>
ozcode::NLayerNetwork::CalculateGradient(arma::mat const& x, arma::mat const& t)
{
	const double loss = CalculateLoss(x, t);
	arma::mat dout = last_layer_->Backward(1);

	for (auto it = layers_.rbegin(); it != layers_.rend(); ++it)
	{
		dout = (*it)->Backward(dout);
	}

	std::map<std::string, arma::mat> grads;

	for (auto it = layers_.cbegin(); it != layers_.cend(); ++it)
	{
		const auto p_affine_layer = dynamic_cast<LayerAffine*>(*it);
		if (p_affine_layer)
		{
			const int layer_index = p_affine_layer->index();
			const std::string w_name = (boost::format("W%1%") % layer_index).str();
			const std::string b_name = (boost::format("b%1%") % layer_index).str();
			grads[w_name] = p_affine_layer->dW();
			grads[b_name] = p_affine_layer->db();
		}
	}

	return grads;
}

void ozcode::NLayerNetwork::SaveWeights(std::string const& dir)
{
	std::filesystem::path out_path = std::filesystem::current_path() / dir;
	if (!std::filesystem::exists(out_path))
		std::filesystem::create_directories(out_path);
	for (std::pair<std::string, arma::mat> p : params_)
	{
		std::filesystem::path path = out_path / (boost::format("%1%.txt") % p.first).str();
		p.second.save(path.string(), arma::arma_ascii, true);
	}
}

double ozcode::NLayerNetwork::GetWeightScale(arma::uword size)
{
	double weight_scale = 0.0;
	switch (weight_init_mechanism_)
	{
	case Sigma:
		weight_scale = 0.01;
		break;
	case Xavier:
		weight_scale = 1.0 / sqrt(double(size));
		break;
	case He:
		weight_scale = sqrt(2.0 / size);
		break;
	default:
		throw std::runtime_error("unsupported weight initial method!");
	}
	return weight_scale;
}

std::vector<std::pair<arma::uword, arma::uword>> ozcode::NLayerNetwork::GetLayerShapes()
{
	std::vector<std::pair<arma::uword, arma::uword>> layer_shapes;
	std::deque<arma::uword> shape_deque(hidden_layer_node_nums_.begin(), hidden_layer_node_nums_.end());
	shape_deque.push_front(input_size_);
	shape_deque.push_back(output_size_);
	for (size_t i = 0; i != shape_deque.size() - 1; ++i)
	{
		std::pair<arma::uword, arma::uword> shape(shape_deque.at(i), shape_deque.at(i + 1));
		layer_shapes.push_back(shape);
	}
	return layer_shapes;
}

void ozcode::NLayerNetwork::free()
{
	for (Layer* layer : layers_)
		delete layer;
	layers_.clear();
	delete last_layer_;
	last_layer_ = nullptr;
}

