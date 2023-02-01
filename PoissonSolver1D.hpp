#ifndef _POISSON_SOLVER_1D_HPP_
#define _POISSON_SOLVER_1D_HPP_

#include "util.hpp"
#include <cufft.h>

template <class T>
class PoissonSolver1D : public CRTP<T,PoissonSolver1D<T>> {

	void solve() { this->self().sol(); }

};

class FFTPoissonSolver1D 
: public PoissonSolver1D<FFTPoissonSolver1D> {

	template<typename InputItor, typename OutputItor>
	void sol(InputItor inBegin, OutputItor outBegin) {
 

	}

};

#endif /* _POISSON_SOLVER_1D_HPP_ */
