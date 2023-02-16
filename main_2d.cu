#include <iostream>
#include <complex>
#include <cmath>
#include <fstream>
#include "FreeStreamSolver.hpp"
#include "Timer.h"
#include "MemSaveReorderCopy.hpp"
#include "PhaseSpaceInitialization.hpp"
#include "DensityReducer.hpp"
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>

using Real = float;
using Complex = std::complex<Real>;

constexpr std::size_t DIM = 4;

constexpr std::size_t nx1 = 100;
constexpr std::size_t nx2 = 80;
constexpr std::size_t nv1 = 66;
constexpr std::size_t nv2 = 60;
constexpr std::size_t nx1Ghost = 4;
constexpr std::size_t nx2Ghost = 4;
constexpr std::size_t nx1Tot = nx1Ghost*2+nx1;
constexpr std::size_t nx2Tot = nx2Ghost*2+nx2;
constexpr std::size_t nTot = nx1Tot*nx2Tot*nv1*nv2;

constexpr Real x1Max =  20, x2Max =  20;
constexpr Real x1Min =  0,  x2Min =  0;
constexpr Real v1Max =  6,  v2Max =  6;
constexpr Real v1Min = -6,  v2Min = -6;

constexpr Real dt = 0.01;


int main(int argc, char* argv[]) {

	Timer timer;

	timer.tick("Asking for GPU memory...");
	cudaSetDevice(1);

	cudaDeviceSetLimit(
		cudaLimitMallocHeapSize, 1048576ULL*1024*3);
	cudaDeviceSetLimit(
		cudaLimitStackSize, 1048576ULL*1024*3);
	cudaDeviceSetLimit(
		cudaLimitPrintfFifoSize, 1048576ULL*1024*3);
	timer.tock();
	
	timer.tick("quakins start...");
	quakins::CoordinateSystem<Real,DIM>
					_coord({nx1,nx2,nv1,nv2},
								 {nx1Ghost,nx2Ghost,0,0},
								 {x1Min,x1Max,x2Min,x2Max,
								  v1Min,v1Max,v2Min,v2Max});

	auto f = [](std::array<Real,DIM> z) {

		auto fx = [](Real x1, Real x2) {
			return std::exp(-std::pow(x1-3,2)
							-std::pow(x2-10,2));	
		};
		auto fv = [](Real v1, Real v2) {
			return std::exp(-std::pow(v1+2,2)/2.
						-std::pow(v2,2)/1.);	
		};

		return fx(z[0],z[1])*fv(z[2],z[3]);
	};
	

	quakins::fbm::FreeStreamSolver<Real,DIM,0> 
					fbmSolverX1(_coord,dt*.5);	
	quakins::fbm::FreeStreamSolver<Real,DIM,1> 
					fbmSolverX2(_coord,dt*.5);

	quakins::MemSaveReorderCopy<Real,DIM,nTot>
					copy0({0,1,3,2},{nx1Tot,nx2Tot,nv1,nv2});
	quakins::MemSaveReorderCopy<Real,DIM,nTot>
					copy1({1,0,3,2},{nx1Tot,nx2Tot,nv2,nv1});
	quakins::MemSaveReorderCopy<Real,DIM,nTot>
					copy2({2,3,1,0},{nx2Tot,nx1Tot,nv1,nv2});
	quakins::MemSaveReorderCopy<Real,DIM,nTot>
					copy3({2,3,1,0},{nv1,nv2,nx1Tot,nx2Tot});


	thrust::device_vector<Real> test1(nTot), test2(nTot);
	thrust::device_vector<Real> dens_e(nx1Tot*nx2Tot), 
															dens_e_buf(nx1Tot*nx2Tot*nv2);

	quakins::DensityReducer<Real,nv1,nx1Tot*nx2Tot*nv2,
		thrust::device_vector> cal_dens_1(v1Min,v1Max);
	quakins::DensityReducer<Real,nv2,nx1Tot*nx2Tot,
		thrust::device_vector> cal_dens_2(v2Min,v2Max);


	timer.tock(); /* quakins start... */

	timer.tick("Phase space initialization...");
	quakins::PhaseSpaceInitialization<Real,DIM> init(&_coord);
	init(test2.begin(),f);
	timer.tock();

	copy0(test2.begin(),test1.begin());
	
	std::ofstream rho_out("rho",std::ios::out);

	std::cout << "main loop start." << std::endl;
	for (std::size_t step=0; step<400; step++) {
		timer.tick("step"+std::to_string(step));	

		fbmSolverX1(test1.begin(),nx1Tot*nx2Tot*nv2);
		
		copy1(test1.begin(),test2.begin());
	
		fbmSolverX2(test2.begin(),nx1Tot*nx2Tot*nv1);
		
		copy2(test2.begin(),test1.begin());
		
		cal_dens_1(test1.begin(),dens_e_buf.begin());
		cal_dens_2(dens_e_buf.begin(),dens_e.begin());
		
		if (step%10==0)
			rho_out << dens_e << std::endl;

		copy3(test1.begin(),test2.begin());
		test1 = test2;
		timer.tock();
	}

	std::ofstream out("df",std::ios::out);
	out << test1 << std::endl;
	
}


