#ifndef PPHASEPICKER_FILTER_H
#define PPHASEPICKER_FILTER_H

#include "butterworth_bandpass.h"

#include <vector>
#include <exception>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>

/**
 * Copy of filtfilt from Matlab R2020b
 * Working only for signal by vector and coeffs A and B by vectors
 * Coeffs A and B must be done by Butterworth bandpass filter
 */

namespace pphase_picker {

    using namespace Eigen;

    template<typename T>
    using VectorX = Vector<T, Dynamic>;

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
        Vector<double, 2 * n + 1> B_;
        Vector<double, 2 * n + 1> A_;
        Vector<double, 2 * n + 1> zi_;

        void getCoeffsAndInitialConditions() {

            constexpr std::size_t nfilt = n * 2 + 1;
            constexpr std::size_t nfact = 3 * (nfilt - 1);

            static const RowVectorXi zero_to_nfilt_minus_two = RowVectorXi::LinSpaced(nfilt - 1, 0, nfilt - 2);
            static const RowVectorXi one_to_nfilt_minus_two = RowVectorXi::LinSpaced(nfilt - 2, 1, nfilt - 2);
            static const RowVectorXi zero_to_nfilt_minus_three = RowVectorXi::LinSpaced(nfilt - 2, 0, nfilt - 3);
            static const RowVectorXi zeros_nfilt_minus_one = RowVectorXi::Zero(nfilt - 1);
            static const RowVectorXi ones_nfilt_minus_two = RowVectorXi::Ones(nfilt - 2);
            static const RowVectorXi minus_ones_nfilt_minus_two = -ones_nfilt_minus_two;
            static const RowVectorXd ones_nfilt_minus_two_d = RowVectorXd::Ones(nfilt - 2);
            static const RowVectorXd minus_ones_nfilt_minus_two_d = -ones_nfilt_minus_two_d;

            static const RowVectorXi rows = (RowVectorXi(nfact - 2) << zero_to_nfilt_minus_two,
                one_to_nfilt_minus_two,
                zero_to_nfilt_minus_three).finished();

            static const RowVectorXi cols = (RowVectorXi(nfact - 2) << zeros_nfilt_minus_one,
                one_to_nfilt_minus_two,
                one_to_nfilt_minus_two).finished();

            static const RowVectorXd vals = (RowVectorXd(nfact - 2) << 1.0 + A_[1],
                A_(seqN(Eigen::fix<2>, Eigen::fix<nfilt - 2>)).transpose(),
                ones_nfilt_minus_two_d,
                minus_ones_nfilt_minus_two_d).finished();

            static const VectorXd rhs = B_(seqN(fix<1>, fix<nfilt - 1>)) - (B_[0] * A_(seqN(fix<1>, fix<nfilt - 1>)));

            static const std::vector<Triplet<double>> triplets = [nfact]() {
                std::vector<Triplet<double>> triplets_first;
                triplets_first.reserve(nfact - 2);
                for (std::size_t i = 0; i < nfact - 2; ++i) {
                    triplets_first.emplace_back(rows[i], cols[i], vals[i]);
                }

                return triplets_first;
            }();

            static const SparseMatrix<double> M = [nfilt]() {
                SparseMatrix<double> M_first(nfilt - 1, nfilt - 1);
                M_first.setFromTriplets(triplets.begin(), triplets.end());
                return M_first;
            }();

            static const BiCGSTAB<SparseMatrix<double>, IncompleteLUT<double> > solver(M);

            zi_ = solver.solve(rhs);
        }

        template<typename T>
        VectorX<T> ffOneChanCat(const VectorX<T>& signal) {
            constexpr std::size_t nfilt = n * 2 + 1;
            constexpr std::size_t nfact = 3 * (nfilt - 1);
            VectorXd zo1 = zi_;
            VectorXd zo2 = zi_;

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
            VectorXd zo1 = zi_;
            VectorXd zo2 = zi_;

            //First filter
            VectorX<T> xt = (-signal(seq(fix<nfact>, fix<1>, fix<-1>))).array() + 2 * signal[0];
            zo1 *= xt[0];
            filter(xt, zo1);
            VectorX<T> Yc2 = filter(signal, zo1);
            xt = (-signal(seq(last - fix<1>, last - fix<nfact>, fix<-1>))).array() + 2 * signal(last);
            VectorX<T> Yc3 = filter(xt, zo1);

            //Second filter
            zo2 *= Yc3(last);
            Yc3.reverseInPlace();
            filter(Yc3, zo2);
            Yc2.reverseInPlace();
            VectorX<T> Yc4 = filter(Yc2, zo2);
            Yc4.reverseInPlace();
            return Yc4;
        }

        template<typename T>
        VectorX<T> filter(const VectorX<T>& signal,
            VectorXd& z) {

            std::size_t input_size = signal.size();
            VectorX<T> Y = VectorX<T>::Zero(input_size);

            std::size_t filter_order = std::max(A_.size(), B_.size());
            std::size_t n_z = z.size();

            z.conservativeResize(filter_order);
            z.tail(filter_order - n_z) << VectorXd::Zero(filter_order - n_z);

            for (std::size_t m = 0; m < input_size; ++m) {
                const T x_m = signal[m];
                Y[m] = B_[0] * x_m + z[0];
                const T y_m = Y[m];
                for (std::size_t i = 1; i < filter_order; ++i) {
                    z[i - 1] = (B_[i] * x_m) + z[i] - (A_[i] * y_m);
                }
            }

            z = z.segment(0, n_z);

            return Y;
        }
    };
}

#endif //PPHASEPICKER_FILTER_H
