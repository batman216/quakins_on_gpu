#ifndef _REORDER_COPY_H_
#define _REORDER_COPY_H_

#include <thrust/iterator/permutation_iterator.h>
#include <thrust/scan.h>
#include <thrust/async/copy.h>
#include <thrust/inner_product.h>
#include "WignerFunction.hpp"


namespace quakins {
	namespace piecewise_reorder_copy {

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

		template<std::size_t dim,bool piter_on_origin>
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
				std::array<std::size_t,dim> idx_m = 
						idxS2M(idx,piter_on_origin ? n_dim_new:n_dim);

				auto p_iter = thrust::make_permutation_iterator
								(idx_m.begin(),order.begin());
				thrust::copy(p_iter,p_iter+dim,idx_m_new.begin());
	
				// take n_dim here, if it is the other way arround, 
				// the permuation_iterator shall point to the new array 
				// instead of the original one as is in this case
				return idxM2S(idx_m_new,piter_on_origin? n_dim:n_dim_new);

			}
		};


		template <template<typename...> typename idx_container,
							typename idx_type,
						 	std::size_t dim,
							typename inputItor,
							typename outputItor,
							bool piter_on_origin
						 >
		class plan {
		
			cudaStream_t *streams;
			idx_container<idx_type> idx_permu; // permutation index
			std::array<std::size_t,dim> n_dim, n_dim_new;
			std::array<std::size_t,dim> order;

			std::size_t n_piece, n_stride;

			inputItor  inBegin;
			outputItor outBegin;
		public:
			plan(cudaStream_t *streams, std::size_t n_piece,
					std::array<std::size_t,dim> n_dim,
					std::array<std::size_t,dim> order,
					inputItor in, outputItor out)
		 		: streams(streams), n_piece(n_piece), 
				  n_dim(n_dim), order(order), inBegin(in), outBegin(out)	{

				auto n_tot = thrust::reduce(n_dim.begin(),n_dim.end(),1,
									thrust::multiplies<std::size_t>());
				n_stride = n_tot / n_piece;
				std::cout << n_stride << std::endl;
				cal_permutation_index<dim, piter_on_origin> op(order,n_dim);

				idx_permu.resize(n_tot);
				thrust::counting_iterator<int> iter(0);
				thrust::transform(iter,iter+n_tot,idx_permu.begin(),op);

			}
				
			void execute(std::size_t i) {
				auto permutationItor = thrust::make_permutation_iterator(
											inBegin,idx_permu.begin());

				thrust::async::copy(thrust::cuda::par.on(streams[i]),
										permutationItor+i*n_stride, 
										permutationItor+(i+1)*n_stride, outBegin);

			}



		}; 

	} // namespace piecewise_reorder_copy

} // namespace quakins 



#endif /* _REORDER_COPY_H_ */
