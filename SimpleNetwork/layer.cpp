#include "pch.h"
#include "layer.h"
#include <regex>
#include <boost/lexical_cast.hpp>

ozcode::Layer::Layer(Layer const& other) : name_(other.name_)
{
}

ozcode::Layer& ozcode::Layer::operator=(Layer const& rhs)
{
	name_ = rhs.name_;
	return *this;
}

ozcode::Layer::Layer(Layer&& other) noexcept : name_(other.name_)
{
	other.name_ = "";
}

ozcode::Layer& ozcode::Layer::operator=(Layer&& rhs) noexcept
{
	if (this != &rhs)
	{
		name_ = rhs.name_;
		rhs.name_ = "";
	}
	return *this;
}

int ozcode::Layer::index() const
{
	const std::regex re("\\d+");
	std::smatch match;
	int index = -1;
	if (std::regex_search(name_, match, re) && match.size() > 0)
		index = boost::lexical_cast<int>(match[0]);
	else
		throw std::runtime_error("layer name no index!");
	return index;
}
