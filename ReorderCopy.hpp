#ifndef _REORDER_COPY_HPP_
#define _REORDER_COPY_HPP_

#include <thrust/iterator/permutation_iterator.h>
#include <thrust/scan.h>
#include <thrust/inner_product.h>
#include "WignerFunction.hpp"

namespace quakins {

	template<std::size_t dim>
	std::size_t idxM2S(std::array<std::size_t,dim> idx_m,
								 		std::array<std::size_t,dim> N) {
		std::array<std::size_t,dim> shift;
		thrust::exclusive_scan(N.begin(),N.end(),shift.begin(),1,
													thrust::multiplies<std::size_t>());

		return thrust::inner_product(idx_m.begin(),idx_m.end(),
										shift.begin(),0);
	}

	template<std::size_t dim>
	std::array<std::size_t,dim> idxS2M(const std::size_t &idx_s,
										const std::array<std::size_t,dim>& N) {
		std::array<std::size_t,dim> idx_m;
		for (std::size_t i=0; i<dim; i++) {
			std::size_t imod = thrust::reduce(N.begin(),N.begin()+i+1,
														1, thrust::multiplies<std::size_t>());
			std::size_t idvd = thrust::reduce(N.begin(),N.begin()+i,
														1, thrust::multiplies<std::size_t>());
			idx_m[i] = (idx_s%imod) / idvd;
		}
		return idx_m;
	}

	template<std::size_t dim>
	struct cal_permutation_index {
		
		const std::array<std::size_t, dim> order, n_dim;
	  std::array<std::size_t,dim> n_dim_new;

		cal_permutation_index(std::array<std::size_t,dim> order,
										std::array<std::size_t,dim> n_dim):
					order(order), n_dim(n_dim) { 
					
			auto p_iter_n = thrust::make_permutation_iterator
							(n_dim.begin(),order.begin());
			thrust::copy(p_iter_n,p_iter_n+dim,n_dim_new.begin());
						
		}

		std::size_t operator()(std::size_t idx) {

			std::array<std::size_t,dim> idx_m_new;
		
			// take n_dim_new here
			std::array<std::size_t,dim> idx_m = idxS2M(idx,n_dim_new);

			auto p_iter = thrust::make_permutation_iterator
							(idx_m.begin(),order.begin());
			thrust::copy(p_iter,p_iter+dim,idx_m_new.begin());
	
			// take n_dim here, if it is the other way arround, 
			// the permuation_iterator shall point to the new array 
			// instead of the original one as is in this case
			return idxM2S(idx_m_new,n_dim);

		}


	};

#if __cplusplus > 201703L
	template <typename T>
	concept Iteratable = requires(T v) { v.begin(); v.end(); };
#endif


	template <typename val_type, 
					 	std::size_t dim,
						template<typename...> typename Container 
					 >
#if __cplusplus > 201703L
	requires Iteratable<Container<val_type>>
#endif
	class ReorderCopy {
	
		std::size_t n_tot;
		std::array<std::size_t,dim> n_dim;
		Container<std::size_t> p_idx;

	public:
		ReorderCopy(std::array<std::size_t,dim> n_dim, 
								std::array<std::size_t,dim> order)
			: n_dim(n_dim) {

			n_tot = thrust::reduce(n_dim.begin(),n_dim.end(),1,
											thrust::multiplies<std::size_t>());
			
			Container<std::size_t> idx_s(n_tot), dim_idx(dim);
			thrust::counting_iterator<int> iter(0);
		
			thrust::copy(iter,iter+idx_s.size(),idx_s.begin());
			thrust::copy(iter,iter+dim,dim_idx.begin());
			
			cal_permutation_index<dim> op(order,n_dim);

			std::cout << "device memory cost for one reorder copy buffer is" 
						<< sizeof(std::size_t)*n_tot/1073741824. 
						<< "G" << std::endl;	
	
			p_idx.resize(n_tot);
			thrust::transform(idx_s.begin(),idx_s.end(),p_idx.begin(),op);

		}

		template<typename OutputItor, typename InputItor>
		void operator()(InputItor inBegin, OutputItor outBegin) {

			auto permutationItor = thrust::make_permutation_iterator(
											inBegin,p_idx.begin());
			thrust::copy(permutationItor, permutationItor+n_tot, outBegin);

		}
	
	};

}



#endif /* _REORDER_COPY_HPP_ */



