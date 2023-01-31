#ifndef _TIMER_H_
#define _TIMER_H_
#include <chrono>
#include <string>
#include <iostream>

class Timer {
		
		std::chrono::time_point<
						std::chrono::high_resolution_clock
						> time1, time2, time3;
public:
		Timer() {}
	
		void tick(std::string message) {
				time1 = std::chrono::system_clock::now();

				std::cout << message << std::flush;		
		}
		void tock() {
					
				time2 = std::chrono::system_clock::now();
				auto int_ms = std::chrono::duration_cast<
								std::chrono::milliseconds>(time2 - time1);
				
				std::cout << "("+ std::to_string(int_ms.count()) +"ms)" << std::endl;
		}


};



#endif /* _TIMER_H_ */
