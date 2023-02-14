#ifndef _MEM_SAVE_REORDER_COPY_HPP_
#define _MEM_SAVE_REORDER_COPY_HPP_

#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scatter.h>
#include "ReorderCopy.hpp"

namespace quakins {

template <typename val_type, 
					std::size_t dim,
				  std::size_t n_tot>
class MemSaveReorderCopy {

	using IntArray = std::array<std::size_t,dim>;
	IntArray order, n_dim;
public:
	MemSaveReorderCopy(IntArray order, IntArray n_dim) 
		: order(order), n_dim(n_dim) {}


	template <typename InputIterator, typename OutputIterator>
	void operator()(InputIterator in_itor_begin, OutputIterator out_itor_begin) {
		

		auto cal_reorder_idx = [l_order=order,l_n_dim=n_dim](int idx_s) {
			// transform i to i'.

			std::array<std::size_t,dim> idx_m;
			idx_m = idxS2M(idx_s,l_n_dim);
			for (std::size_t i=0; i<dim; i++) {
				std::size_t imod = thrust::reduce(l_n_dim.begin(),l_n_dim.begin()+i+1,
															1, thrust::multiplies<std::size_t>());
				std::size_t idvd = thrust::reduce(l_n_dim.begin(),l_n_dim.begin()+i,
															1, thrust::multiplies<std::size_t>());
				idx_m[i] = (idx_s%imod) / idvd;
			}

			// reorder the multi-indices
			auto pitor = thrust::make_permutation_iterator(
															l_n_dim.begin(),l_order.begin());
			auto idx_pitor = thrust::make_permutation_iterator(
															idx_m.begin(),l_order.begin());


			std::array<std::size_t,dim> shift;	
			thrust::exclusive_scan(pitor,pitor+dim,shift.begin(),1,
													thrust::multiplies<std::size_t>());

			return thrust::inner_product(idx_pitor,idx_pitor+dim,
										shift.begin(),0);

		};

		auto titor_begin = thrust::make_transform_iterator(
						thrust::make_counting_iterator(0), cal_reorder_idx);
	
		thrust::scatter(in_itor_begin, in_itor_begin+n_tot,
											titor_begin, out_itor_begin);

	}


};

}




#endif /* _MEM_SAVE_REORDER_COPY_HPP_ */
