#ifndef COMMON_H
#define COMMON_H

#define _USE_MATH_DEFINES

#include <iostream>
#include <fstream>
#include <Eigen/Dense>

namespace phase_picker
{
	using namespace Eigen;

	template <typename T>
	using idx_t = typename VectorX<T>::Index;

}

#endif //COMMON_H