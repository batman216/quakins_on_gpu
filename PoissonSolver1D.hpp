#ifndef _POISSON_SOLVER_1D_HPP_
#define _POISSON_SOLVER_1D_HPP_

#include "util.hpp"
#include <cufft.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace quakins {

template <class T>
class PoissonSolver1D : public CRTP<T,PoissonSolver1D<T>> {
	
public:
	template <typename in_type,typename out_type,
					  template<typename...> typename Container>
	void operator()(const Container<in_type>& density, 
									Container<out_type>& potential) { 
		this->self().solve(in_begin, out_begin); 
	}

};

template <typename val_type,
					template<typename...> typename Container>
class FFTPoissonSolver1D 
: public PoissonSolver1D<FFTPoissonSolver1D<val_type,
												Container>> {

	cufftHandle plan_fwd, plan_inv;
	Container<val_type> inv_k_square, buffer;
	val_type L;
	std::size_t n, nBd;

public:
	FFTPoissonSolver1D(std::size_t n, std::size_t nBd, val_type L)
	: n(n), nBd(nBd), L(L) {
		cufftCreate(&plan_fwd);
		cufftCreate(&plan_inv);
		cufftPlan1d(&plan_fwd,n,CUFFT_R2C,1);
		cufftPlan1d(&plan_inv,n,CUFFT_C2R,1);
	
		inv_k_square.resize(n);
		buffer.resize(n);
	}

	template <typename in_type, typename out_type>
	void operator()(const Container<in_type>& dens, 
									Container<out_type>& pot) { 

		auto rho_ptr = (cufftReal*) thrust::raw_pointer_cast(dens.data()+nBd);
		auto buf_ptr = (cufftComplex*) thrust::raw_pointer_cast(buffer.data());
		auto phi_ptr = (cufftReal*) thrust::raw_pointer_cast(pot.data()+nBd);
		
		cufftExecR2C(plan_fwd,rho_ptr,buf_ptr);

		// k-space
		int nint = static_cast<int>(n);
		val_type dk = 2.*M_PI/L;

		auto titor = thrust::make_transform_iterator(
			thrust::make_counting_iterator(0),
		[dk,nint](int idx) {
			val_type k_value = dk*static_cast<val_type>
														(idx<nint/2 ? idx:(idx-nint));
			return idx==0? 0. :
							1./k_value/k_value/static_cast<val_type>(nint);
		});

		std::cout << titor[0] << " ";
		thrust::transform(titor,titor+n,buffer.begin(),
											buffer.begin(),thrust::multiplies<val_type>());
		
		cufftExecC2R(plan_inv,buf_ptr,phi_ptr);

	}

};

} // namespace quakins

#endif /* _POISSON_SOLVER_1D_HPP_ */
