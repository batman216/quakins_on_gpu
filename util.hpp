#ifndef _UTIL_HPP_
#define _UTIL_HPP_

#include <iostream>
#include <string>
#include <map>
#include <memory>


// dynamic polyorphism
template <typename t_Product> 
class Factory {
public:
	virtual t_Product *produce() = 0;
protected:
	Factory() {}	
	virtual ~Factory() {}
};

/// forward declaration
template <typename t_Product> class Proxy; 
	
template <typename t_Product, typename t_ConcreteProduct> 
class ConcreteFactory : public Factory<t_Product> {
public:
	explicit ConcreteFactory(std::string p_name) {
		Proxy<t_Product>::Instance().registerProduct(this,p_name);
	}
	t_Product *produce() { return new t_ConcreteProduct; }

};

/**
 *   The class Proxy is like a flyweight factory
 */
template <typename t_Product>
class Proxy {
	Proxy() {}  /// private creator for singleton
	Proxy(const Proxy& other) {}  
public:
	/**
	 *  The client fetch/purchase concrete product by p_name
	 *  according to the regedit
	 */
	std::map<std::string, Factory<t_Product>*> regedit;
	
	static Proxy<t_Product>& Instance() {
		static Proxy<t_Product> instance;
		/// Meyers Singleton: create an instance only when this function is called.
		return instance;
	}
	void registerProduct(Factory<t_Product>* reg, std::string p_name) {
		regedit[p_name] = std::move(reg);
	}

	t_Product* get(std::string p_name) { /// flyweight singleton
		if (regedit.find(p_name) != regedit.end())
			return regedit[p_name]->produce();
			/// produce	
		else {
			std::cout << "no product named " << p_name 
					<< "registered." << std::endl;	
			return nullptr;
		}
	}
};



// static polymorphism via CRTP copied from
// https://gist.github.com/12ff54e/7643d7361d7221e4d3d0918ec3e193d6
// @Dr. Zhong Qi

template <class T, class...>
struct CRTP {
    T& self() { return static_cast<T&>(*this); }
    const T& self() const { return static_cast<const T&>(*this); }
};




template<typename T, 
		template<typename...> typename Container>
concept isThrustContainer =
	std::same_as<Container<T>,thrust::host_vector<T>>
		|| std::same_as<Container<T>,thrust::device_vector<T>>;

template<typename T, 
		template<typename...> typename Container>
requires isThrustContainer<T,Container>
std::ostream& operator<<(std::ostream& os, 
		const Container<T>& vec) {
	thrust::copy(vec.begin(),vec.end(),
		std::ostream_iterator<T>(os," "));
	return os;
}


template<std::size_t dim>
std::size_t idxM2S(std::array<std::size_t,dim> idx_m,
							 		std::array<std::size_t,dim> N) {
	std::array<std::size_t,dim> shift;
	thrust::exclusive_scan(N.begin(),N.end(),shift.begin(),1,
												thrust::multiplies<std::size_t>());
	return thrust::inner_product(idx_m.begin(),idx_m.end(),
									shift.begin(),0);
}

template<std::size_t dim>
std::array<std::size_t,dim> idxS2M(const std::size_t &idx_s,
									const std::array<std::size_t,dim>& N) {
	std::array<std::size_t,dim> idx_m;
	for (std::size_t i=0; i<dim; i++) {
		std::size_t imod = thrust::reduce(N.begin(),N.begin()+i+1,
													1, thrust::multiplies<std::size_t>());
		std::size_t idvd = thrust::reduce(N.begin(),N.begin()+i,
													1, thrust::multiplies<std::size_t>());
		idx_m[i] = (idx_s%imod) / idvd;
	}
	return idx_m;
}


#endif /* _UTIL_HPP_ */
