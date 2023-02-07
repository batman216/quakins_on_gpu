#ifndef _INTEGRATOR_HPP_
#define _INTEGRATOR_HPP_

#include <thrust/inner_product.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <array>

#include <iostream>
namespace quakins {

template <typename val_type,
				  std::size_t num,
					template<typename...> typename Container
				 >
class Integrator : public thrust::unary_function
			<typename Container<val_type>::iterator,val_type> {

	Container<val_type> coeff_array;

public:

	using Iterator = Container<val_type>::iterator; 

	Integrator(val_type a, val_type b)	{

		coeff_array.resize(num);

		thrust::counting_iterator<unsigned int> cit(0);

		val_type C = (b-a)/3./static_cast<val_type>(num);
		thrust::transform(cit,cit+num,coeff_array.begin(),
		[C](int idx) {
			return 	idx%2==0 ? 2.*C : 4.*C;			
		});
		coeff_array[0] = 1.*C;

	}

	val_type operator()(Iterator valBegin) {

		return thrust::inner_product(coeff_array.begin(),
										coeff_array.end(), valBegin, 0.);

	}

};

template<typename val_type, std::size_t n, std::size_t n_stride,
				template<typename...> typename Container>
struct DensityReducer {
	
	using itor_type = Container<val_type>::iterator;
	Container<std::size_t> begin_indices;

	DensityReducer() {
	
		begin_indices.resize(n);	
		for (std::size_t i=0; i<n; i++) 
			begin_indices[i] = i*n_stride;


	}

	template <typename IntegralFunctor>
	void operator()(IntegralFunctor integral, 
									itor_type f_begin, itor_type dens_begin) {
		
		auto pitor_begin = thrust::make_permutation_iterator(
										f_begin, begin_indices);
		auto zitor_begin = thrust::make_zip_iterator(
										thrust::make_tuple(pitor_begin,dens_begin));

		thrust::for_each(zitor_begin,zitor_begin+n,
						[&](auto &tuple) {
			thrust::get<1>(tuple) = integral(get<0>(tuple));
		});

	}

};



} // namespace quakins



#endif /* _INTEGRATOR_HPP_ */
