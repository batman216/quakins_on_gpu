#include <iostream>
#include <complex>
#include <cmath>
#include <fstream>
#include "FreeStreamSolver.hpp"
#include "PoissonSolver1D.hpp"
#include "Timer.h"
#include "ReorderCopy.hpp"
#include "reorder_copy.h"
#include "DensityReducer.hpp"
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

template<typename T, 
		template<typename...> typename Container>
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

	timer.tick("creating host reorder copy...");
	quakins::ReorderCopy<Real,2, true,
				thrust::host_vector> copy_h2d(_wf.N,{0,1});
	quakins::ReorderCopy<Real,2, false, 
				thrust::host_vector> copy_d2h
								({nx1Tot,nv1},{0,1});
	timer.tock();
	
		
	timer.tick("creating device reorder copy...");
	quakins::ReorderCopy<Real,2, true,
					thrust::device_vector> copy_d2d_1
									({nx1Tot,nv1},{1,0});
	quakins::ReorderCopy<Real,2, true,
					thrust::device_vector> copy_d2d_2
									({nv1,nx1Tot},{1,0});
	timer.tock();
	
	thrust::device_vector<Real> test1(_wf.nTot);
	thrust::device_vector<Real> test2(_wf.nTot);
	copy_h2d(_wf.begin(),test1.begin());

	thrust::device_vector<Real> dens_e(nx1Tot), dens_i(nx1Tot);

	quakins::DensityReducer<Real,nv1,nx1Tot,
					thrust::device_vector> cal_dens(v1Min,v1Max);

	std::ofstream rho_out("rho",std::ios::out);

	std::cout << "main loop start." 
	<< std::endl; for (int step=0; step<500; step++) {

		timer.tick("step"+std::to_string(step));
		fbmSolverX1(test1.begin(),nx1Tot);
		copy_d2d_1(test1.begin(),test2.begin());

		cal_dens(test2.begin(), dens_e.begin());

		rho_out << dens_e;

		copy_d2d_2(test2.begin(),test1.begin());
		timer.tock();
	}

	std::ofstream out("df",std::ios::out);
	thrust::copy(test1.begin(), test1.end(),
							std::ostream_iterator<Real>(out," "));
	out << std::endl;
	
}







