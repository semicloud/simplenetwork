#include "pch.h"
#include "layer.h"
#include <regex>
#include <boost/lexical_cast.hpp>

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
