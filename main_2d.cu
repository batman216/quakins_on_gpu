#include <iostream>
#include <complex>
#include <cmath>
#include <fstream>
#include "FreeStreamSolver.hpp"
#include "Timer.h"
#include "MemSaveReorderCopy.hpp"
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>

using Real = double;
using Complex = std::complex<Real>;
using VecH = thrust::host_vector<Real>;
using VecD = thrust::device_vector<Real>;


constexpr std::size_t nx1 = 220;
constexpr std::size_t nx2 = 206;
constexpr std::size_t nv1 = 82;
constexpr std::size_t nv2 = 80;
constexpr std::size_t nx1Ghost = 4;
constexpr std::size_t nx2Ghost = 2;
constexpr std::size_t nx1Tot = nx1Ghost*2+nx1;
constexpr std::size_t nx2Tot = nx2Ghost*2+nx2;
constexpr std::size_t nTot = nx1Tot*nx2Tot*nv1*nv2;

constexpr Real x1Max =  20, x2Max =  20;
constexpr Real x1Min =  0,  x2Min =  0;
constexpr Real v1Max =  6,  v2Max =  6;
constexpr Real v1Min = -6,  v2Min = -6;

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
	cudaSetDevice(1);

	cudaError_t err = cudaDeviceSetLimit(
					cudaLimitMallocHeapSize, 1048576ULL*1024*3);
	err = cudaDeviceSetLimit(
					cudaLimitStackSize, 1048576ULL*1024*4);
	err = cudaDeviceSetLimit(
					cudaLimitPrintfFifoSize, 1048576ULL*1024*1);
		

	Timer timer;
	
	quakins::CoordinateSystemHost<Real,4>
					_coord({nx1,nx2,nv1,nv2},
								 {nx1Ghost,nx2Ghost,0,0},
								 {x1Min,x1Max,x2Min,x2Max,
								  v1Min,v1Max,v2Min,v2Max});
	quakins::WignerFunctionHost<Real,4> 
			_wf({nx1Tot,nx2Tot,nv1,nv2});

	auto f = [](std::array<Real,4> z) {

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
	timer.tick("initializing...");
	quakins::init(_coord,_wf,f); timer.tock();
	quakins::fbm::FreeStreamSolver<Real,4,0> 
					fbmSolverX1(_wf,_coord,dt*.5);	
	quakins::fbm::FreeStreamSolver<Real,4,1> 
					fbmSolverX2(_wf,_coord,dt*.5);	

		
	quakins::MemSaveReorderCopy<Real,4,nTot>
					copy0({0,1,3,2},{nx1Tot,nx2Tot,nv1,nv2});
	quakins::MemSaveReorderCopy<Real,4,nTot>
					copy1({1,0,3,2},{nx1Tot,nx2Tot,nv2,nv1});
	quakins::MemSaveReorderCopy<Real,4,nTot>
					copy2({1,0,3,2},{nx2Tot,nx1Tot,nv1,nv2});

	thrust::device_vector<Real> test1(nTot), test2(nTot);
	test2 = _wf.hVec;

	copy0(test2.begin(),test1.begin());
	
	std::ofstream bout("dfbegin",std::ios::out);
	//bout << test2 << std::endl;


	std::cout << "main loop start." << std::endl;
	for (int step=0; step<800; step++) {
		timer.tick("step"+std::to_string(step));	

		fbmSolverX1(test1.begin(),nx1Tot*nx2Tot*nv2);
		
		copy1(test1.begin(),test2.begin());
	
		fbmSolverX2(test2.begin(),nx1Tot*nx2Tot*nv1);
		
		copy2(test2.begin(),test1.begin());
		
		timer.tock();
	}

	std::ofstream out("df",std::ios::out);
	out << test1 << std::endl;
	
}


