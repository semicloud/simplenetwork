#ifndef COLLECTION_UTILS_H
#define COLLECTION_UTILS_H

#include <map>
#include <vector>

/**
 * This is a helper class for collections in c++
 * Author: Li Qinyong
 * Company: Software Institute Chinese Academy of Science
 *
 */
namespace ozcode {

	/**
	 * \brief This function retrieve keys as std::vector from map
	 * \tparam K the type of key
	 * \tparam V the type of value
	 * \param map map object
	 * \return keys of the map
	 */
	template <typename K, typename V>
	std::vector<K> GetKeys(std::map<K, V> const& map) {
		std::vector<K> ans;
		ans.reserve(map.size());
		for (auto const& imap : map) ans.push_back(imap.first);
		return ans;
	}

	/**
	 * \brief This function retrieve values as std::vector from map
	 * \tparam K K the type of key
	 * \tparam V V the type of value
	 * \param map map object
	 * \return values of the map
	 */
	template <typename K, typename V>
	std::vector<K> GetValues(std::map<K, V> const& map) {
		std::vector<V> ans;
		ans.reserve(map.size());
		for (auto const& imap : map) ans.push_back(imap.second);
		return ans;
	}

}  // namespace ozcode

#endif
