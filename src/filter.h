#ifndef FILTER_H
#define FILTER_H

#include "common.h"

#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "butterworth_bandpass.h"

/**
 * Copy of filtfilt from Matlab R2020b
 * Working only for signal by vector and coeffs_ A and B by vectors
 * Coeffs A and B must be done by Butterworth bandpass filter
 */

namespace phase_picker {

    template<std::size_t n>
    class FiltFiltButter {
    public:
        FiltFiltButter() = default;

        FiltFiltButter(double* b, double* a)
        {
            Map<Vector<double, 2 * n + 1>> B_map(b);

            Map<Vector<double, 2 * n + 1>> A_map(a);
            B_ = B_map;
            A_ = A_map;

            const double a0 = A_[0];
            if (a0 != 1.0) {
                A_ /= a0;
                B_ /= a0;
            }

            getCoeffsAndInitialConditions();
        }

        static FiltFiltButter<n> create_butterworth_bandpass_filter(double dt,
                                                                    double flp,
                                                                    double fhp) {
            const auto fnq = 1.0 / (2.0 * dt);

            const auto uhf = fhp / fnq;
            const auto lhf = flp / fnq;

            auto p_picker_coeffs = bwbp_args(n, lhf, uhf);

            return FiltFiltButter<n>(p_picker_coeffs->ccof, p_picker_coeffs->dcof);
        }

        template<typename T>
        VectorX<T> filtfilt(const VectorX<T>& signal) {

            std::size_t len = signal.size();

            if (len < 10000) {
                return ffOneChanCat(signal);
            }
            else {
                return ffOneChan(signal);
            }
        }
    private:

        static constexpr std::size_t nfilt = n * 2 + 1;
        static constexpr std::size_t nfact = 3 * (nfilt - 1);

        template <std::size_t size>
        using RowVectorNi = Matrix<int, 1, size>;

        template <std::size_t size>
        using RowVectorNd = Matrix<double, 1, size>;

        Vector<double, nfilt> B_;
        Vector<double, nfilt> A_;
        Vector<double, nfilt-1> zi_;

        void getCoeffsAndInitialConditions() {

            const RowVectorNi<nfilt-1> zero_to_nfilt_minus_two = RowVectorNi<nfilt-1>::LinSpaced(0, nfilt - 2);
            const RowVectorNi<nfilt - 2> one_to_nfilt_minus_two = RowVectorNi<nfilt - 2>::LinSpaced(1, nfilt - 2);
            const RowVectorNi<nfilt - 2> zero_to_nfilt_minus_three = RowVectorNi<nfilt - 2>::LinSpaced(0, nfilt - 3);
            const RowVectorNi<nfilt-1> zeros_nfilt_minus_one = RowVectorNi<nfilt-1>::Zero();
            const RowVectorNi<nfilt - 2> ones_nfilt_minus_two = RowVectorNi<nfilt - 2>::Ones();
            const RowVectorNi<nfilt - 2> minus_ones_nfilt_minus_two = -ones_nfilt_minus_two;
            const RowVectorNd<nfilt - 2> ones_nfilt_minus_two_d = RowVectorNd<nfilt - 2>::Ones();
            const RowVectorNd<nfilt - 2> minus_ones_nfilt_minus_two_d = -ones_nfilt_minus_two_d;

            const RowVectorNi<nfact-2> rows = (RowVectorNi<nfact-2>{} << zero_to_nfilt_minus_two,
                one_to_nfilt_minus_two,
                zero_to_nfilt_minus_three).finished();

            const RowVectorNi<nfact-2> cols = (RowVectorNi<nfact-2>{} << zeros_nfilt_minus_one,
                one_to_nfilt_minus_two,
                one_to_nfilt_minus_two).finished();

            const RowVectorNd<nfact-2> vals = (RowVectorNd<nfact-2>{} << 1.0 + A_[1],
                A_(seqN(Eigen::fix<2>, Eigen::fix<nfilt - 2>)).transpose(),
                ones_nfilt_minus_two_d,
                minus_ones_nfilt_minus_two_d).finished();

            const Vector<double, nfilt-1> rhs = B_(seqN(fix<1>, fix<nfilt - 1>)) - (B_[0] * A_(seqN(fix<1>, fix<nfilt - 1>)));

            Matrix<double, nfilt-1, nfilt-1> M = Matrix<double, nfilt-1, nfilt-1>::Zero();

            for (auto i = 0; i < nfact-2; ++i) {
                M(rows[i], cols[i]) = vals[i];
            }

            const JacobiSVD<Matrix<double, nfilt-1, nfilt-1>, FullPivHouseholderQRPreconditioner> solver(M, ComputeFullU | ComputeFullV);

            zi_ = solver.solve(rhs);
        }

        template<typename T>
        VectorX<T> ffOneChanCat(const VectorX<T>& signal) {
            constexpr std::size_t nfilt = n * 2 + 1;
            constexpr std::size_t nfact = 3 * (nfilt - 1);
            VectorX<T> zo1 = zi_.template cast<T>();
            VectorX<T> zo2 = zi_.template cast<T>();

            VectorX<T> Y_temp(2 * nfact + signal.size());
            Y_temp << (-signal(seq(fix<nfact>, fix<1>, fix<-1>))).array() + 2 * signal[0],
                signal,
                (-signal(seq(last - fix<1>, last - fix<nfact>, fix<-1>))).array() + 2 * signal(last);
            zo1 *= Y_temp[0];
            Y_temp = filter(Y_temp, zo1);
            zo2 *= Y_temp(last);
            Y_temp.reverseInPlace();
            Y_temp = filter(Y_temp, zo2);

            return Y_temp(seqN(fix<nfact>, signal.size()).reverse());
        }

        template<typename T>
        VectorX<T> ffOneChan(const VectorX<T>& signal) {
            constexpr std::size_t nfilt = n * 2 + 1;
            constexpr std::size_t nfact = 3 * (nfilt - 1);
            VectorX<T> zo1 = zi_.template cast<T>();
            VectorX<T> zo2 = zi_.template cast<T>();

            //First filter
            VectorX<T> xt = (-signal(seq(fix<nfact>, fix<1>, fix<-1>))).array() + 2 * signal[0];
            zo1 *= xt[0];
            filter(xt, zo1);
            VectorX<T> Yc2 = filter(signal, zo1);
            xt = (-signal(seq(last - fix<1>, last - fix<nfact>, fix<-1>))).array() + 2 * signal(last);
            VectorX<T> Yc3 = filter(xt, zo1);

            //Second filter
            zo2 *= Yc3(last);
            filter((VectorX<T>)Yc3.reverse(), zo2);
            VectorX<T> Yc4 = filter((VectorX<T>)Yc2.reverse(), zo2).reverse();
            return Yc4;
        }

        template<typename T>
        VectorX<T> filter(const VectorX<T>& signal,
            VectorX<T>& z) {

            std::size_t input_size = signal.size();
            VectorX<T> Y = VectorX<T>::Zero(input_size);

            std::size_t filter_order = std::max(A_.size(), B_.size());
            std::size_t n_z = z.size();

            z.conservativeResize(filter_order);
            z.tail(filter_order - n_z) << VectorX<T>::Zero(filter_order - n_z);

            for (std::size_t m = 0; m < input_size; ++m) {
                const T x_m = signal[m];
                Y[m] = B_[0] * x_m + z[0];
                const T y_m = Y[m];
                Vector<double, nfilt> tmp_coeffs = (B_ * x_m) - (A_ * y_m);
                for (std::size_t i = 1; i < filter_order; ++i) {
                    z[i - 1] = tmp_coeffs[i] + z[i];
                }
            }

            z = z.segment(0, n_z);

            return Y;
        }
    };
}

#endif //FILTER_H
