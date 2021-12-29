#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include "common.h"

#include <vector>
#include <Eigen/Dense>

namespace phase_picker
{
    enum class window_function_t
    {
        BOXCAR,
        GAUSSIAN,
        HAMMING,
        HANN,
        HANNING,
        KONNO_OHMACHI,
        PARZEN,
        TRIANG,
        DEFAULT
    };
    
    struct math_utils
    {
        template<typename T>
        static void detrend(const VectorX<T>& data,
                        VectorX<T> &detrend_data) {
            using to_T = T;
            std::size_t n = data.size();
            const VectorX<T> x_axis = VectorX<T>::LinSpaced(n, to_T(0), to_T(n-1));
            T sum_x = static_cast<T>((n - 1) * n / 2);
            T sum_x2 = static_cast<T>((n - 1) * n * (2 * n - 1) / 6);
            T sum_y = data.sum();
            T sum_xy = data.dot(x_axis);

            T a = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
            T b = (sum_y - a * sum_x) / n;

            detrend_data = (data - (a * x_axis)).array() - b;
        }

        template<typename T>
        static VectorXi function_hisotgram(const VectorX<T> &y,
            const std::size_t nbins) {
        
            using to_T = T;
            auto y_max = y.maxCoeff();
            auto y_min = y.minCoeff() - std::numeric_limits<T>::epsilon();

            VectorXi idx = ceil((to_T(nbins) * ((y.array() - y_min) / (y_max - y_min))).array()).template cast<int>();

            VectorXi histogram = VectorXi::Zero(nbins);
            for (const auto& id : idx) {
                if (id != 0 && id < nbins + 1)
                {
                    histogram[id-1] += 1;
                }
                
            }

            return histogram;
        }

        template <typename T>
        static auto find_local_maximums(const VectorX<T> &f) -> std::vector<std::pair<T, idx_t<T>>>
        {
            std::vector<std::pair<T, idx_t<T> >> peaks;
            peaks.reserve(f.size());
            for (auto i = 1; i < f.size()-1; ++i)
            {
                if (f[i-1] <= f[i] && f[i+1] <= f[i])
                {
                    peaks.emplace_back(std::make_pair(f[i], i));
                }
            }

            return peaks;
        }

        template <typename T>
        static VectorX<T> vector_gradient(const VectorX<T> &v,
            T h = T(1.0))
        {
            auto v_size = v.size();
            auto rev_h = T(1.0) / h;
            auto half_rev_h = rev_h / T(2.0);

            VectorX<T> g = VectorX<T>::Zero(v_size);

            if (v_size == 1)
            {
                g << v[0];
            } else if (v_size == 2)
            {
                g << ((v[1] - v[0]) * rev_h), ((v[2] - v[1]) * rev_h);
            } else 
            {
                g << ((v[1] - v[0]) * rev_h),
                    (((v.segment(2, v_size-2) - v.head(v_size-2)) / T(2.0)) * half_rev_h),
                    ((v(last) - v(last-1)) * rev_h);
            }
            
            return g;
        }

        template <typename U, typename T>
        static VectorXd polyfit(const VectorX<U> &x,
            const VectorX<T>& y,
            unsigned long order) {
            
            auto x_size = x.size();
            auto x_mean = x.mean() / x_size;
            VectorX<U> x_without_mean = x.array() - x_mean;
            
            U x_std = std::sqrt((U(1.0) / static_cast<U>(x_size-1)) * x_without_mean.squaredNorm());
            
            VectorX<U> x_normalized = x_without_mean.array() / x_std;
            
            // Construct Vandermonde matrix
            Matrix<U, Dynamic, Dynamic> V_matrix(x_size, order+1);
            V_matrix.col(order) = VectorX<U>::Ones(x_size);
            
            for (auto i = order; i > 0; --i) {
                V_matrix.col(i-1) = V_matrix.col(i).cwiseProduct(x_normalized);
            }

            // method 1) SOLE solution by QR-decomposition
            
            VectorXd p = V_matrix.colPivHouseholderQr().solve(y).cast<double>();

            // method 2) Estimated coeffs_ by OLS estimation
            /*auto v_matrix_transpose = v_matrix.transpose();
            auto p = (((v_matrix_transpose * v_matrix).reverse() * v_matrix_transpose) * y).cast<double>();*/

            return p;
            
        }

        template <typename T>
        static VectorX<T> integrate(const VectorX<T> &f,
            const double dt) {
            auto size = f.size();
            VectorX<T> F = VectorX<T>::Zero(size);
            auto half_dt = dt / 2.0;
            
            for (auto i = 1; i < size; ++i) {
                F[i] = F[i-1] + (f[i-1] + f[i])*half_dt;
            }

            return F;
        }

        template <typename T>
        static VectorX<T> smooth_curve(const VectorX<T> &x,
                                unsigned long window_size,
                                window_function_t method,
                                double b_coeff = 20.0) {
            using to_T = T;
            window_size = (window_size % 2) ? window_size : window_size - 1;
            auto halfw = (window_size / 2) + 1;
            VectorX<T> W;
            
            switch (method)
            {
            case window_function_t::BOXCAR:
                {
                    //Boxcar window
                    W = VectorX<T>::Ones(window_size);
                    break;
                }
            case window_function_t::GAUSSIAN:
                {
                    //Gaussian window
                    auto L = window_size - 1;
                    W = exp((-0.5 * (2.5 * (VectorX<T>::LinSpaced(window_size, to_T(0), to_T(L)).array() - to_T(L / 2)) / to_T(L / 2)).cwiseAbs2()).array());
                    break;
                }
            case window_function_t::HAMMING:
                {
                    //Hamming window
                    auto L = window_size - 1;
                    W = 0.54 - (0.46 * cos((2 * M_PI * VectorX<T>::LinSpaced(window_size, to_T(0), to_T(L)).array() / to_T(L)).array())).array();
                    break;
                }
            case window_function_t::HANN:
                {
                    //Hann window
                    auto L = window_size - 1;
                    W = 0.5 - (0.5 * cos((2 * M_PI * VectorX<T>::LinSpaced(window_size, to_T(0), to_T(L)).array() / to_T(L)).array())).array();
                    break;
                }
            case window_function_t::HANNING:
                {
                    W = 0.5 * (1 - cos((2 * M_PI * VectorX<T>::LinSpaced(window_size, to_T(1), to_T(window_size)).array() / to_T(window_size)).array()));
                    break;
                }
            case window_function_t::KONNO_OHMACHI:
                {
                    VectorX<T> one_to_window_size_seq = VectorX<T>::LinSpaced(window_size, to_T(1), to_T(window_size));
                    VectorX<T> tmp = b_coeff * (one_to_window_size_seq.array() / to_T(halfw)).log10();
                    W = (sin(tmp.array()) / tmp.array()).unaryExpr([](const T& v) { return std::pow(v, 4); });
                    W[halfw-1] = to_T(1.0);
                    break;
                }
            case window_function_t::PARZEN:
                {
                    auto L = window_size - 1;
                    long halfL = L / 2;
                    VectorX<T> k = VectorX<T>::LinSpaced(window_size, -to_T(halfL), to_T(halfL));
                    VectorX<T> k1 = (k.array() < -to_T(halfL / 2)).select(k, 0);
                    VectorX<T> k2 = (abs(k.array()).array() <= to_T(halfL / 2)).select(k, 0);
                    VectorX<T> w1 = 2 * cube((1 - abs(k1.array()).array() / halfL).array());
                    VectorX<T> k2_div_halfL = abs(k2.array()).array() / halfL;
                    VectorX<T> w2 = 1 - 6*square(k2_div_halfL.array())+ 6*cube(k2_div_halfL.array());
                    W = w1 + w2 + w1.reverse();
                    break;
                }
            case window_function_t::TRIANG:
                {
                    auto L = window_size + 1;
                    VectorX<T> W_tmp = 2 * (VectorX<T>::LinSpaced(L / 2, to_T(1), to_T(L / 2)).array() / to_T(L));
                    W.resize(window_size);
                    W << W_tmp, W_tmp.head((window_size - 1) / 2).reverse();
                    break;
                }
            default:
                {
                    return x;
                }
            }

            W /= W.sum();

            auto N = x.size();
            auto d = window_size - 1;
            VectorX<T> arg = VectorX<T>::Zero(N + d);
            VectorX<T> x_pad(2 * d + N);
            static const VectorX<T> d_zeros = VectorX<T>::Zero(d);
            x_pad << d_zeros, x, d_zeros;
            W.reverseInPlace();
            for (auto i = 0; i < N + d; ++i)
            {
                arg[i] = W.dot(x_pad.segment(i, window_size));
            }

            return arg.segment(halfw - 1, N);
        }
    };
}

#endif // !MATH_UTILS_H