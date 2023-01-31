#include <iostream>
#include <complex>
#include <cmath>
#include <fstream>
#include "FreeStreamSolver.hpp"
#include "Timer.h"
#include "ReorderCopy.hpp"
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>

using Real = float;
using Complex = std::complex<Real>;
using VecH = thrust::host_vector<Real>;
using VecD = thrust::device_vector<Real>;


constexpr std::size_t nx1 = 56;
constexpr std::size_t nx2 = 50;
constexpr std::size_t nv1 = 42;
constexpr std::size_t nv2 = 40;
constexpr std::size_t nx1Ghost = 2;
constexpr std::size_t nx2Ghost = 2;
constexpr std::size_t nx1Tot = nx1Ghost*2+nx1;
constexpr std::size_t nx2Tot = nx2Ghost*2+nx2;

constexpr Real x1Max =  20, x2Max =  20;
constexpr Real x1Min =  0,  x2Min =  0;
constexpr Real v1Max =  6,  v2Max =  6;
constexpr Real v1Min = -6,  v2Min = -6;

constexpr Real dt = 0.01;

template<typename T>
std::ostream& operator<<(std::ostream& os, 
		const thrust::host_vector<T>& vec) {
	thrust::copy(vec.begin(),vec.end(),
		std::ostream_iterator<T>(os," "));
	return os;
}

int main(int argc, char* argv[]) {

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
			return std::exp(-std::pow(v1-2,2)/2.
						-std::pow(v2,2)/2.);	
		};

		return fx(z[0],z[1])*fv(z[2],z[3]);
	};
	timer.tick("initializing...");
	quakins::init(_coord,_wf,f); timer.tock();
	
	std::ofstream bout("dfbegin",std::ios::out);
	bout << _wf.hVec << std::endl;

	quakins::fbm::FreeStreamSolver<Real,4,0> 
					fbmSolverX1(_wf,_coord,dt*.5);	
	quakins::fbm::FreeStreamSolver<Real,4,1> 
					fbmSolverX2(_wf,_coord,dt*.5);	


	timer.tick("requst memory on GPU...");
	thrust::device_vector<Real> test1(_wf.nTot);
	thrust::device_vector<Real> test2(_wf.nTot);
	timer.tock(); 

	timer.tick("creating host reorder copy...");
	quakins::ReorderCopy<Real,4, 
					thrust::host_vector> copy_h2d(_wf.N,{0,1,3,2});
	timer.tock();
	
	timer.tick("transfering data from host to device...");
	copy_h2d(_wf.begin(),test1.begin());
	timer.tock();
		
	timer.tick("creating device reorder copy...");
	quakins::ReorderCopy<Real,4, 
					thrust::device_vector> copy_d2d_1
									({nx1Tot,nx2Tot,nv2,nv1},{1,0,3,2});
	quakins::ReorderCopy<Real,4, 
					thrust::device_vector> copy_d2d_2
									({nx2Tot,nx1Tot,nv1,nv2},{1,0,3,2});
	timer.tock();


	std::cout << "main loop start." << std::endl;
	for (int step=0; step<300; step++) {
		timer.tick("step"+std::to_string(step));	
		fbmSolverX1(test1.begin(),nx1Tot*nx2Tot*nv2);
		copy_d2d_1(test1.begin(),test2.begin());
		fbmSolverX2(test2.begin(),nx1Tot*nx2Tot*nv1);
		copy_d2d_2(test2.begin(),test1.begin());
		timer.tock();
	}
	
	timer.tick("transfering data from device to host...");
  _wf.hVec = test1;
	timer.tock();

	std::ofstream out("df",std::ios::out);
	out << _wf.hVec << std::endl;
	
}


