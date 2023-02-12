#ifndef _FREE_STREAM_SOLVER_HPP_
#define _FREE_STREAM_SOLVER_HPP_
#include "WignerFunction.hpp"
#include "CoordinateSystem.hpp"

#include <thrust/tuple.h>
#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <fstream>

namespace quakins {
namespace fbm {

template <typename Iterator>
class strided_chunk_range
{
    public:

    typedef typename thrust::iterator_difference<Iterator>::type difference_type;

    struct stride_functor : public thrust::unary_function<difference_type,difference_type>
    {
        difference_type stride;
        int chunk;
        stride_functor(difference_type stride, int chunk)
            : stride(stride), chunk(chunk) {}

        __host__ __device__
        difference_type operator()(const difference_type& i) const
        {
            int pos = i/chunk;
            return ((pos * stride) + (i-(pos*chunk)));
        }
    };

    typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
    typedef typename thrust::transform_iterator<stride_functor, CountingIterator> TransformIterator;
    typedef typename thrust::permutation_iterator<Iterator,TransformIterator>     PermutationIterator;

    // type of the strided_range iterator
    typedef PermutationIterator iterator;

    // construct strided_range for the range [first,last)
    strided_chunk_range(Iterator first, Iterator last, difference_type stride, int chunk)
        : first(first), last(last), stride(stride), chunk(chunk) {assert(chunk<=stride);}

    iterator begin(void) const
    {
        return PermutationIterator(first, TransformIterator(CountingIterator(0), stride_functor(stride, chunk)));
    }

    iterator end(void) const
    {
        int lmf = last-first;
        int nfs = lmf/stride;
        int rem = lmf-(nfs*stride);
        return begin() + (nfs*chunk) + ((rem<chunk)?rem:chunk);
    }

    protected:
    Iterator first;
    Iterator last;
    difference_type stride;
    int chunk;
};

template <typename val_type, std::size_t dim, std::size_t ndim>
struct FreeStreamSolver {
	
	std::size_t nx, nv, nBd, nTot, vdim;
	thrust::device_vector<val_type> Phi; 		// flux function
	thrust::device_vector<val_type> alpha;	// shift length
	val_type h;  // spactial interval

	FreeStreamSolver(const WignerFunctionHost<val_type,dim>& wf,
	const CoordinateSystemHost<val_type,dim>& coord,val_type dt) {

		nBd  = coord.nBd[ndim];
		nx   = coord.nz[ndim];
		vdim = ndim + (dim>>1);
		nv   = coord.nz[vdim];
		h    = coord.dz[ndim];
		nTot = wf.nTot;
		
		// prepare the flux function
		thrust::host_vector<val_type> _Phi(wf.nTot);
		Phi  = _Phi; //to device

		// calculate shift wihtin dt
		thrust::host_vector<val_type> _alpha(nv);
		thrust::transform(coord.coord[vdim].begin(),
			coord.coord[vdim].end(),_alpha.begin(),
		 	[&](auto &v){ return v*dt/h; 	}); // v<0
		
			alpha = _alpha;  // to device
	}
			
	template <typename itor_type>
	void operator()(itor_type itor_begin, std::size_t n_chunk) {
				
		// the outermost dimension is calculated sequentially
		std::size_t n_step = nTot/n_chunk; 
		
		// Boundary Condition
		strided_chunk_range<itor_type> 
			left_inside(itor_begin+nBd,itor_begin+nTot,nx+2*nBd, nBd);
		strided_chunk_range<itor_type> 
			left_outside(itor_begin,itor_begin+nTot,nx+2*nBd, nBd);
		strided_chunk_range<itor_type> 
			right_inside(itor_begin+nx,itor_begin+nTot,nx+2*nBd, nBd);
		strided_chunk_range<itor_type> 
			right_outside(itor_begin+nx+nBd,itor_begin+nTot,nx+2*nBd, nBd);

		thrust::copy(left_inside.begin(),left_inside.end(),
										right_outside.begin());
		thrust::copy(right_inside.begin(),right_inside.end(),
										left_outside.begin());


		// calculate the flux function \Phi
		auto zitor_pos_begin 
			= make_zip_iterator(thrust::make_tuple(
																			Phi.begin() + nTot/2,
																			itor_begin-1 + nTot/2,
																			itor_begin + nTot/2,  
																			itor_begin+1 +nTot/2));
		auto zitor_neg_begin 
			=	make_zip_iterator(thrust::make_tuple(
																			Phi.begin(),
																			itor_begin,
																			itor_begin+1,  
																			itor_begin+2));

		for (std::size_t i = 0; i<n_step/2; i++) {

			thrust::for_each(zitor_neg_begin+nBd-1,
											zitor_neg_begin+n_chunk-nBd+1,
			[a=alpha[i]](auto tuple){
				thrust::get<0>(tuple) = a*(thrust::get<2>(tuple) 
				-(1-a)*(1+a)/6*(thrust::get<3>(tuple)-thrust::get<2>(tuple))
				-(2+a)*(1+a)/6*(thrust::get<2>(tuple)-thrust::get<1>(tuple)));
			});
			zitor_neg_begin += n_chunk;
		} // v < 0
	
		for (std::size_t i = n_step/2; i<n_step; i++) {

			thrust::for_each(zitor_pos_begin+nBd-1,
											zitor_pos_begin+n_chunk-nBd+1,
										[a=alpha[i]](auto tuple){
				thrust::get<0>(tuple) = a*(thrust::get<2>(tuple) 
				+(1-a)*(2-a)/6*(thrust::get<3>(tuple)-thrust::get<2>(tuple))
				+(1-a)*(1+a)/6*(thrust::get<2>(tuple)-thrust::get<1>(tuple)));
			});
			zitor_pos_begin += n_chunk;
		} // v > 0

		auto zitor_begin = thrust::make_zip_iterator(thrust::make_tuple(
														itor_begin,Phi.begin()-1,Phi.begin()));

		// calculate f[i](t+dt)=f[i](t) + Phi[i-1/2] -Phi[i+1/2]
		thrust::for_each(zitor_begin+nBd,zitor_begin
										+nTot-nBd, [](auto tuple) {
			thrust::get<0>(tuple) += thrust::get<1>(tuple)
															-thrust::get<2>(tuple);
		});
	}
};

} // namespace fbm
} // namespace quakins


#endif /* _FREE_STREAM_SOLVER_HPP_ */
