#ifndef _DENSITY_REDUCER_HPP_
#define _DENSITY_REDUCER_HPP_

#include <thrust/inner_product.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <array>

#include <iostream>

namespace quakins {

template<typename val_type, std::size_t n, std::size_t n_batch,
				template<typename...> typename Container>
struct DensityReducer {

	const val_type coeff;

	DensityReducer(val_type a, val_type b) :
		coeff((b-a)/3./static_cast<val_type>(n)) {}

	template <typename itor_type>
	void operator()(itor_type f_begin, itor_type dens_begin) {

		val_type C = this->coeff;
		auto zitor_begin = thrust::make_zip_iterator(
												thrust::make_tuple(
													thrust::make_counting_iterator(0),f_begin));

		auto titor_begin = make_transform_iterator(
												zitor_begin,
												[C](auto _tuple){ return 
													static_cast<val_type>(
														thrust::get<1>(_tuple) *
														(thrust::get<0>(_tuple)%2==0? 2.*C:4.*C)); });

		auto binary_pred = [](int i,int j) {
			return i/n == j/n;
		};

		thrust::reduce_by_key(thrust::make_counting_iterator(0),
													thrust::make_counting_iterator(
																	static_cast<int>(n*n_batch)),
													titor_begin,
													thrust::make_discard_iterator(),
													dens_begin,
													binary_pred);

	}

};



} // namespace quakins



#endif /* _DENSITY_REDUCER_HPP_ */
