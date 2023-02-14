#include <iostream>
#include <complex>
#include <cmath>
#include <fstream>
#include "FreeStreamSolver.hpp"
#include "PoissonSolver1D.hpp"
#include "Timer.h"
#include "ReorderCopy.hpp"
#include "MemSaveReorderCopy.hpp"
#include "reorder_copy.h"
#include "DensityReducer.hpp"
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>

using Real = float;
using Complex = std::complex<Real>;


constexpr std::size_t nx1 = 500;
constexpr std::size_t nv1 = 256;
constexpr std::size_t nx1Ghost = 6;
constexpr std::size_t nx1Tot = nx1Ghost*2+nx1;
constexpr std::size_t nTot = nx1Tot*nv1;

constexpr Real x1Max =  20;
constexpr Real x1Min =  0;
constexpr Real v1Max =  6;
constexpr Real v1Min = -6;

constexpr Real dt = 0.01;

template<typename T, 
		template<typename...> typename Container>
concept isAcontainer = requires (Container<T>& a) {
	a.begin(); a.end();
};

template<typename T, 
		template<typename...> typename Container>
requires isAcontainer<T,Container>
std::ostream& operator<<(std::ostream& os, 
		const Container<T>& vec) {
	thrust::copy(vec.begin(),vec.end(),
		std::ostream_iterator<T>(os," "));
	return os;
}

int main(int argc, char* argv[]) {

	Timer timer;
	
	quakins::CoordinateSystemHost<Real,2>
					_coord({nx1,nv1}, {nx1Ghost,0},
								 {x1Min,x1Max, v1Min,v1Max});
	quakins::WignerFunctionHost<Real,2> 
			_wf({nx1Tot,nv1});

	auto f = [](std::array<Real,2> z) {

		auto fx = [](Real x1) {
			return 1.+.1*std::cos(2.*M_PI/x1Max*x1);
		};
		auto fv = [](Real v1) {
			return std::exp(-std::pow(v1,2)/2.)/std::sqrt(2.*M_PI);
		};

		return fx(z[0])*fv(z[1]);
	};

	timer.tick("initializing...");
	quakins::init(_coord,_wf,f); timer.tock();
	
	std::ofstream bout("dfbegin",std::ios::out);
	bout << _wf.hVec << std::endl;

	quakins::fbm::FreeStreamSolver<Real,2,0> 
					fbmSolverX1(_wf,_coord,dt*.5);	

	
	thrust::device_vector<Real> 
		ion(_wf.nTot), ion_buf(_wf.nTot),
		electron(_wf.nTot), electron_buf(_wf.nTot);

	thrust::device_vector<Real> 
		dens_e(nx1Tot), dens_i(nx1Tot), potential(nx1Tot);

	quakins::DensityReducer<Real,nv1,nx1Tot,
					thrust::device_vector> cal_dens(v1Min,v1Max);

	quakins::FFTPoissonSolver1D<Real,
					thrust::device_vector> solvePoisson(nx1,nx1Ghost,x1Max-x1Min);

	quakins::MemSaveReorderCopy<Real,2,nTot> copy1({1,0},{nx1Tot,nv1});
	quakins::MemSaveReorderCopy<Real,2,nTot> copy2({1,0},{nv1,nx1Tot});

	std::ofstream rho_out("rho",std::ios::out);
	std::ofstream phi_out("phi",std::ios::out);

	electron = _wf.hVec;

	std::cout << "main loop start." << std::endl;
	for (int step=0; step<100; step++) {
		
		timer.tick("step"+std::to_string(step));
		for (int ie=0; ie<10; ie++) {

			fbmSolverX1(electron.begin(),nx1Tot);

			copy1(electron.begin(),electron_buf.begin());
			cal_dens(electron_buf.begin(), dens_e.begin());

			solvePoisson(dens_e,potential);

			copy2(electron_buf.begin(),electron.begin());
		}
		timer.tock();
		rho_out << dens_e;
		phi_out << potential;
	}

	std::ofstream out("df",std::ios::out);
	out << electron << std::endl;
}




