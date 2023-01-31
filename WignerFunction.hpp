#ifndef _WIGNER_FUNCTION_HPP_
#define _WIGNER_FUNCTION_HPP_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <iostream>
#include <iterator>
#include <cstddef>
#include "CoordinateSystem.hpp"

namespace quakins {

	template <typename val_type, std::size_t dim>
	struct WignerFunctionHost {
		
		thrust::host_vector<val_type> hVec;
		
		WignerFunctionHost(std::array<std::size_t,dim> N) 
			: N(N) {
			nTot = thrust::reduce(N.begin(),N.end(), 1,
					thrust::multiplies<std::size_t>());
			std::cout << "Host memory cost for the "
				<< dim << "d array is about " << 
					sizeof(val_type)*nTot/1073741824. 
						<< "G" << std::endl;	
			hVec.resize(nTot);
			

			// calculatet the interval between two data for each dimension
			dim_shift[0] = 1;
			for (std::size_t i=1; i<dim; i++)
				dim_shift[i] = thrust::reduce(N.begin(),
					N.begin()+i-1,1,thrust::multiplies<std::size_t>());
		}
	
		auto begin() { return hVec.begin(); }
		auto end()   { return hVec.end(); }
	
		const auto begin() const { return hVec.begin(); }
		const auto end() const { return hVec.end(); }
	
		// # each dimension
		std::array<std::size_t,dim> N;         

		// interval between two data for each dimension
		std::array<std::size_t,dim> dim_shift; 
		
		// total number n0*n1*n2*n3*...
		std::size_t nTot;

		const auto operator()(std::array<std::size_t,dim> idx_multi) const {
			
			return hVec[thrust::inner_product(idx_multi.begin(),
											idx_multi.end(),dim_shift.begin(),0.)];
		}


		auto& operator()(std::array<std::size_t,dim> idx_multi) {

			return hVec[thrust::inner_product(idx_multi.begin(),
											idx_multi.end(),dim_shift.begin(),0.)];
		}



	};


	template <typename val_type, std::size_t dim>
	struct WignerFunctionDevice {
	
		thrust::device_vector<val_type> dVec;
		
		WignerFunctionDevice( 
										WignerFunctionHost<val_type,dim> &wfh) 
		: N(wfh.N), nTot(wfh.nTot) {
//			dVec.resize(wfh.nTot);
	//		thrust::copy(wfh.hVec.begin(),wfh.hVec.end(),dVec.begin());
	      dVec = wfh.hVec;
		}
	
		auto begin() { return dVec.begin(); }
		auto end()   { return dVec.end(); }
		const auto begin() const { return dVec.begin(); }
		const auto end() const { return dVec.end(); }
	
		std::array<std::size_t,dim> N;
		std::size_t nTot;

	};

	template <typename val_type, 
					  std::size_t dim,
						typename f_init>
	void init(const CoordinateSystemHost<val_type,dim>& coord,
						WignerFunctionHost<val_type,dim>& wf, f_init f) {

		thrust::counting_iterator<std::size_t> countItor(0);
		thrust::transform(countItor,countItor+wf.nTot,
				wf.begin(), [&](std::size_t idx ) { 	return f(coord[idx]); });	
	
	}


} // namespace quakins


#endif /* _WIGNER_FUNCTION_HOST_H_ */
