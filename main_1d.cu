#include <iostream>
#include <complex>
#include <cmath>
#include <fstream>
#include "FreeStreamSolver.hpp"
#include "Timer.h"
#include "ReorderCopy.hpp"
#include "reorder_copy.h"
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>

using Real = float;
using Complex = std::complex<Real>;
using VecH = thrust::host_vector<Real>;
using VecD = thrust::device_vector<Real>;


constexpr std::size_t nx1 = 508;
constexpr std::size_t nv1 = 256;
constexpr std::size_t nx1Ghost = 2;
constexpr std::size_t nx1Tot = nx1Ghost*2+nx1;

constexpr Real x1Max =  20;
constexpr Real x1Min =  0;
constexpr Real v1Max =  6;
constexpr Real v1Min = -6;

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
	
	quakins::CoordinateSystemHost<Real,2>
					_coord({nx1,nv1}, {nx1Ghost,0},
								 {x1Min,x1Max, v1Min,v1Max});
	quakins::WignerFunctionHost<Real,2> 
			_wf({nx1Tot,nv1});

	auto f = [](std::array<Real,2> z) {

		auto fx = [](Real x1) {
			return std::exp(-std::pow(x1-3,2));
		};
		auto fv = [](Real v1) {
			return std::exp(-std::pow(v1-2,2)/2.);
		};

		return fx(z[0])*fv(z[1]);
	};
	timer.tick("initializing...");
	quakins::init(_coord,_wf,f); timer.tock();
	
	std::ofstream bout("dfbegin",std::ios::out);
	bout << _wf.hVec << std::endl;

	quakins::fbm::FreeStreamSolver<Real,2,0> 
					fbmSolverX1(_wf,_coord,dt*.5);	


	timer.tick("requst memory on GPU...");
	thrust::device_vector<Real> test1(_wf.nTot);
	timer.tock(); 

	timer.tick("creating host reorder copy...");
	quakins::ReorderCopy<Real,2, true,
				thrust::host_vector> copy_h2d(_wf.N,{0,1});
	quakins::ReorderCopy<Real,2, false, 
				thrust::host_vector> copy_d2h
								({nx1Tot,nv1},{0,1});
	timer.tock();
	
		
	timer.tick("creating device reorder copy...");
	quakins::ReorderCopy<Real,2, true,
					thrust::device_vector> copy_d2d
									({nx1Tot,nv1},{0,1});
	timer.tock();

	copy_h2d(_wf.begin(), test1.begin());	
	std::cout << "main loop start." << std::endl;

	for (int step=0; step<300; step++) {
	//		timer.tick("step"+std::to_string(step));	
		fbmSolverX1(test1.begin(),nx1Tot);
	//	timer.tock();
	}

	copy_d2h(test1.begin(),_wf.begin());
	std::ofstream out("df",std::ios::out);
	out << _wf.hVec << std::endl;
	
}


