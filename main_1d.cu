#include <iostream>
#include <complex>
#include <cmath>
#include <fstream>
#include "FreeStreamSolver.hpp"
#include "PoissonSolver1D.hpp"
#include "Timer.h"
#include "MemSaveReorderCopy.hpp"
#include "DensityReducer.hpp"
#include "PhaseSpaceInitialization.hpp"
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

constexpr Real dt = (x1Max-x1Min)/nx1/v1Max/2.3;



int main(int argc, char* argv[]) {
	cudaSetDevice(1);
	std::cout << "dt=" << dt << std::endl;
	Timer timer;
	
	quakins::CoordinateSystem<Real,2>
					_coord({nx1,nv1}, {nx1Ghost,0},
								 {x1Min,x1Max, v1Min,v1Max});

	auto f = [](std::array<Real,2> z) -> Real {

		auto fx = [](Real x1) {
			return 1.+.1*std::cos(2.*M_PI/x1Max*x1);
		};
		auto fv = [](Real v1) {
			return std::exp(-std::pow(v1,2)/2.)/std::sqrt(2.*M_PI);
		};

		return static_cast<Real>(fx(z[0])*fv(z[1]));
	};


	quakins::fbm::FreeStreamSolver<Real,2,0> 
					fbmSolverX1(_coord,dt*.5);	
	
	thrust::device_vector<Real> 
		ion(nTot), ion_buf(nTot),
		electron(nTot), electron_buf(nTot);
	
	timer.tick("Phase space initialization...")
	quakins::PhaseSpaceInitialization<Real,2> init(&_coord);
	init(electron.begin(),f);
	timer.tock();

	std::ofstream bout("dfbegin",std::ios::out);
	bout << electron << std::endl;


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




