#ifndef _COORDINATE_SYSTEM_HPP_
#define _COORDINATE_SYSTEM_HPP_

#include <array> 
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/for_each.h>
#include <thrust/reduce.h> 

namespace quakins {

template <typename val_type, std::size_t dim>
struct CoordinateSystem {
	
	CoordinateSystem(std::array<std::size_t,dim> nz,
											 std::array<std::size_t,dim> nBd,
											 std::array<val_type,dim*2> range)
		: nz(nz), nBd(nBd), range(range)	{
		for (std::size_t i=0; i<dim; i++)
			dz[i] = (range[2*i+1]-range[2*i])
								/static_cast<val_type>(nz[i]);
			
		for (std::size_t i=0; i<dim; i++) {
			nzTot[i] = nz[i] + 2*nBd[i];
			coord[i].resize(nzTot[i]);
			thrust::sequence(coord[i].begin()+nBd[i],coord[i].end()-nBd[i]);
			thrust::for_each(coord[i].begin()+nBd[i],coord[i].end()-nBd[i],
											[&](val_type& z){ z = z*dz[i]+range[2*i]; });

		}
	}	
	
	std::array<thrust::host_vector<val_type>,dim> coord;
	std::array<std::size_t,dim> nz, nzTot, nBd;
	std::array<val_type,dim> dz;
	std::array<val_type,dim*2> range;
		
	const std::array<val_type,dim>
	operator[](std::size_t idx) const {
		std::array<val_type,dim> z;
		for (std::size_t i=0; i<dim; i++) {

			std::size_t dvd_num = thrust::reduce(nzTot.begin(),
											nzTot.begin()+i, 1,
												thrust::multiplies<std::size_t>());
			std::size_t mod_num = dvd_num * nzTot[i];
			z[i] = coord[i][(idx%mod_num)/dvd_num];
		}
		return z;
		
	}


};
	

} // namespace quakins



#endif /* _COORDINATE_SYSTEM_HPP_ */
