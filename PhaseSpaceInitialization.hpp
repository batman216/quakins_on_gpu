#ifndef _PHASE_SPACE_INITIALIZATION_HPP_
#define _PHASE_SPACE_INITIALIZATION_HPP_

#include "CoordinateSystem.hpp"
#include "util.hpp"

namespace quakins {


template <typename val_type, std::size_t dim>
class PhaseSpaceInitialization {

	CoordinateSystem<val_type,dim> *coord;

public:
	PhaseSpaceInitialization(CoordinateSystem<val_type,dim> *coord)
		: coord(coord) {}

	
	template<typename Iterator, class DistFunc>
	void operator()(Iterator itor_begin, DistFunc f) {
		
		auto n_dim = coord->nzTot;
		auto range = coord->range;
		auto dz = coord->dz;
		int nTot = thrust::reduce(n_dim.begin(), n_dim.end(),1,
																thrust::multiplies<std::size_t>());

		auto trans_op = [f,n_dim,range,dz](int idx) {
			auto idx_m = idxS2M(idx,n_dim);
			std::array<val_type,dim> co;
			for (std::size_t i=0; i<dim; i++)
				co[i] = range[i*2] +idx_m[i]*dz[i];
			return f(co);
		};
		auto titor = thrust::make_transform_iterator(
									thrust::make_counting_iterator(0),trans_op);
	
		thrust::copy(titor,titor+nTot,itor_begin);


	}

};

} // namespace quakins

#endif /* _PHASE_SPACE_INITIALIZATION_HPP_ */







