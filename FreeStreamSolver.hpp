#ifndef _FREE_STREAM_SOLVER_HPP_
#define _FREE_STREAM_SOLVER_HPP_
#include "WignerFunction.hpp"
#include "CoordinateSystem.hpp"

#include <thrust/tuple.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <fstream>

namespace quakins {
	namespace fbm {

		template <typename val_type, std::size_t dim, std::size_t ndim>
		struct FreeStreamSolver {
			
			typedef thrust::
							counting_iterator<std::size_t> CountingIterator;
			typedef typename thrust::
							device_vector<val_type>::iterator ValIterator;
			typedef thrust::tuple<ValIterator, 
								ValIterator, ValIterator, ValIterator> PhiIteratorTuple;
			typedef thrust::tuple<ValIterator, 
								ValIterator, ValIterator> IteratorTuple;
			typedef thrust::zip_iterator<PhiIteratorTuple> PhiZipIterator;
			typedef thrust::zip_iterator<IteratorTuple> ZipIterator;

			std::size_t nx, nv, nBd, nTot, vdim;
			thrust::device_vector<val_type> Phi; 		// flux function
			thrust::device_vector<val_type> alpha;	// shift length
			val_type h;

			FreeStreamSolver(const 
											WignerFunctionHost
											<val_type,dim>& wf,
											 const 
											CoordinateSystemHost
											<val_type,dim>& coord,
											val_type dt) {
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

			void operator()(thrust::device_vector<val_type>::iterator iter_begin,
											std::size_t n_chunk) {
				std::size_t n_step = nTot/n_chunk;
				std::cout << n_step << std::endl;
				PhiZipIterator phiZipItorPosBegin(thrust::make_tuple(
													Phi.begin() + nTot/2,
													iter_begin-1 + nTot/2,
													iter_begin + nTot/2,  
													iter_begin+1 +nTot/2  
													));
				PhiZipIterator phiZipItorNegBegin(thrust::make_tuple(
													Phi.begin(),
													iter_begin,
													iter_begin+1,  
													iter_begin+2  
													));
				
				for (std::size_t i = 0; i<n_step/2; i++) {

					thrust::for_each(phiZipItorNegBegin+nBd,
													phiZipItorNegBegin+n_chunk-nBd,
					[a=alpha[i]](auto tuple){
						thrust::get<0>(tuple) = a*(thrust::get<2>(tuple) 
						-(1-a)*(1+a)/6*(thrust::get<3>(tuple)-thrust::get<2>(tuple))
						-(2+a)*(1+a)/6*(thrust::get<2>(tuple)-thrust::get<1>(tuple)));
					});
					phiZipItorNegBegin += n_chunk;
				}
				for (std::size_t i = n_step/2; i<n_step; i++) {

					thrust::for_each(phiZipItorPosBegin+nBd,
													phiZipItorPosBegin+n_chunk-nBd,
												[a=alpha[i]](auto tuple){
						thrust::get<0>(tuple) = a*(thrust::get<2>(tuple) 
						+(1-a)*(2-a)/6*(thrust::get<3>(tuple)-thrust::get<2>(tuple))
						+(1-a)*(1+a)/6*(thrust::get<2>(tuple)-thrust::get<1>(tuple)));
					});
					phiZipItorPosBegin += n_chunk;
				}

				ZipIterator zipIteratorBegin(thrust::make_tuple(
																iter_begin,Phi.begin()-1,Phi.begin()));

				thrust::for_each(zipIteratorBegin+nBd,zipIteratorBegin
												+nTot-nBd, [](auto tuple) {
					thrust::get<0>(tuple) += thrust::get<1>(tuple)
																	-thrust::get<2>(tuple);					
				});

			}
		};
	}
} // namespace quakins


#endif /* _FREE_STREAM_SOLVER_HPP_ */
