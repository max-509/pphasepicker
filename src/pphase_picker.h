#ifndef _PPHASE_PICKER
#define _PPHASE_PICKER

#define _USE_MATH_DEFINES

#include "butterworth_bandpass.h"
#include "filter.h"

#include <cstddef>
#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>
#include <limits>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

namespace pphase_picker {

	using namespace Eigen;

	template<typename T>
	using VectorX = Vector<T, Dynamic>;

	struct pphase_picker_result {
		double pphase_time_arrival = -1.0;
		double snr_decibel = -1.0;
	};

	enum class waveform_type {
		SM, WM, NA
	};

	enum class function_jump_finder_type : long {
		HIST, SIGMA
	};

	template<typename Func>
	struct lambda_as_visitor_wrapper : Func {
		explicit lambda_as_visitor_wrapper(const Func& f) : Func(f) {}
		template<typename S, typename I>
		void init(const S& v, I i, I j) { return Func::operator()(v, i, j); }
	};

	template<typename Mat, typename Func>
	void visit_lambda(const Mat& m, const Func& f) {
		lambda_as_visitor_wrapper<Func> visitor(f);
		m.visit(visitor);
	}

	class PphasePicker {
	public:
		PphasePicker(double dt, bool is_filter, double flp, double fhp) : dt_(dt), is_filter_(is_filter) {
			if (is_filter) {
				init_butterworth_bandpass_filter(flp, fhp, dt);
			}
		}

		template<typename T>
		pphase_picker_result calculate(const Map<VectorX<T>>& x,
			double undamped_period,
			double damping_ratio,
			function_jump_finder_type finder_type,
			std::size_t bin_size,
			const bool to_peak) {

			pphase_picker_result result{};

			VectorX<T> signal = x;

			if (is_filter_) {
				T inf_norm_inverse = 1.0 / signal.template lpNorm<Infinity>();
				signal *= inf_norm_inverse;
				signal = filter.filtfilt<T>(signal);
			}
			detrend(signal, signal);

			typename VectorX<T>::Index end_signal;

			if (to_peak) {
				T max_val = signal.cwiseAbs().maxCoeff(&end_signal);
			}
			else {
				end_signal = signal.size() - 1;
			}
			end_signal += 1;

			double omega_n = 2.0 * M_PI / undamped_period;
			double C = 2.0 * damping_ratio * omega_n;
			double K = std::pow(omega_n, 2);

			Matrix2d A;
			A << 0.0, 1.0,
				-K, -C;

			static const Matrix2d eye_2x2 = Matrix2d::Identity();
			static const Vector2d zero_one = (Vector2d() << 0, 1).finished();

			Matrix<T, Dynamic, Dynamic> y(2, end_signal);
			y.col(0) << 0.0, 0.0;
			Matrix2d Ae = (A * dt_).exp();
			Vector2d AeB = (A.inverse() * (Ae - eye_2x2)) * zero_one;

			for (typename VectorX<T>::Index i = 1; i < end_signal; ++i) {
				y.col(i) = (Ae * y.col(i - 1)) + (AeB * signal[i]);
			}


			VectorX<T> velocity = y.row(1).transpose();
			VectorX<T> d_energy_damping = C * velocity.cwiseAbs2();

			double loc = -1.0;
			typename VectorX<T>::Index pick;
			switch (finder_type) {
			case pphase_picker::function_jump_finder_type::HIST:
				pick = get_pick_hist(d_energy_damping, signal, bin_size);

				if (pick == -1) {
					pick = get_pick_hist(d_energy_damping,
						signal,
						std::ceil(static_cast<double>(bin_size) / 2.0));
					if (pick != -1) {
						loc = static_cast<double>(pick + 1) * dt_;
					}
				}
				else {
					loc = static_cast<double>(pick + 1) * dt_;
				}
				break;
			case pphase_picker::function_jump_finder_type::SIGMA:
				pick = get_pick_sigma(d_energy_damping);
				if (pick != -1) {
					loc = static_cast<double>(pick + 1) * dt_;
				}
				break;
			default:
				break;
			}

			double snr = -1.0;
			if (pick != -1) {
				VectorX<T> noise = x.segment(0, pick + 1);
				snr = get_snr(x, noise);
			}

			result.pphase_time_arrival = loc;
			result.snr_decibel = snr;

			return result;
		}

		~PphasePicker() {
			free(coeffs);
		}
	private:
		double dt_;
		bool is_filter_;
		butter_bandpass_coefficients* coeffs = nullptr;
		FiltFiltButter<4> filter;

		void init_butterworth_bandpass_filter(double flp,
			double fhp,
			double dt) {
			double fnq = 1.0 / (2.0 * dt);

			double uhf = fhp / fnq;
			double lhf = flp / fnq;

			coeffs = bwbp_args(4, lhf, uhf);

			filter = FiltFiltButter<4>(coeffs->ccof, coeffs->dcof);
		}

		template<typename T>
		void detrend(const VectorX<T>& data,
							VectorX<T> &detrend_data) {
			std::size_t n = data.size();
			const VectorX<T> x_axis = VectorX<T>::LinSpaced(n, 0, n-1);
			T sum_x = ((n - 1) * n) / 2;
			T sum_x2 = ((n - 1) * n * (2 * n - 1)) / 6;
			T sum_y = data.sum();
			T sum_xy = data.dot(x_axis);

			T a = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
			T b = (sum_y - a * sum_x) / n;

			detrend_data = (data - (a * x_axis)).array() - b;
		}

		template<typename T>
		Vector2d statelevel(const VectorX<T>& y,
			const std::size_t nbins) {
			auto y_max = y.maxCoeff();
			auto y_min = y.minCoeff() - std::numeric_limits<T>::epsilon();

			VectorXi idx = ((nbins * (y.array() - y_min)) / (y_max - y_min)).ceil().template cast<int>();

			VectorXi histogram = VectorXi::Zero(nbins);
			for (const int& id : idx) {
				histogram[id-1] += 1;
			}

			y_min = y_min + std::numeric_limits<T>::epsilon();
			auto Ry = y_max - y_min;
			auto dy = Ry / nbins;

			typename VectorXi::Index i_low = -1;
			for (typename VectorXi::Index i = 0; i < histogram.size(); ++i) {
				if (histogram[i] > 0) {
					i_low = i;
					break;
				}
			}
			typename VectorXi::Index i_high = -1;
			for (typename VectorXi::Index i = histogram.size() - 1; i >= 0; --i) {
				if (histogram[i] > 0) {
					i_high = i;
					break;
				}
			}

			typename VectorXi::Index l_low = i_low;
			typename VectorXi::Index l_high = i_low + std::floor(static_cast<double>(i_high - i_low) / 2.0);
			typename VectorXi::Index u_low = l_high;
			typename VectorXi::Index u_high = i_high;
			
			VectorXi l_hist = histogram.segment(l_low, l_high - l_low + 1);
			VectorXi u_hist = histogram.segment(u_low, u_high - u_low + 1);

			Vector2d levels = Vector2d::Zero();

			typename VectorXi::Index i_max_l_hist, i_max_u_hist;
			l_hist.maxCoeff(&i_max_l_hist);
			u_hist.maxCoeff(&i_max_u_hist);

			levels[0] = y_min + dy * (static_cast<double>(l_low + i_max_l_hist + 1) - 0.5);
			levels[1] = y_min + dy * (static_cast<double>(u_low + i_max_u_hist + 1) - 0.5);

			return levels;
		}

		template<typename T>
		typename VectorX<T>::Index get_pick_hist(const VectorX<T>& d_energy_damping,
			const VectorX<T>& signal,
			std::size_t bin_size) {

			Vector2d R = statelevel(d_energy_damping, bin_size);
			auto R_0 = R[0];

			typename VectorX<T>::Index loc_i = -1;
			for (typename VectorX<T>::Index i = 0; i < d_energy_damping.size(); ++i) {
				if (d_energy_damping[i] > R_0) {
					loc_i = i;
					break;
				}
			}

			typename VectorX<T>::Index pick = -1;
			VectorX<T> signal_cross = signal.segment(0, loc_i).cwiseProduct(signal.segment(1, loc_i));
			for (typename VectorX<T>::Index i = signal_cross.size() - 1; i >= 0; --i) {
				if (signal_cross[i] < 0) {
					pick = i;
					break;
				}
			}

			return pick;
		}

		template <typename T>
		typename VectorX<T>::Index get_pick_sigma(const VectorX<T>& d_energy_damping) {
			auto n_d_energy_damping = d_energy_damping.size();
			T d_energy_damping_mean = d_energy_damping.mean();
			T d_energy_damping_std = std::sqrt((d_energy_damping.squaredNorm() / n_d_energy_damping) - std::pow(d_energy_damping_mean, 2));

			typename VectorX<T>::Index p_wave_begin = -1;

			for (typename VectorX<T>::Index i = 0; i < n_d_energy_damping; ++i) {
				if (std::abs(d_energy_damping_mean - d_energy_damping[i]) > d_energy_damping_std) {
					p_wave_begin = i;
					break;
				}
			}

			return p_wave_begin;
		}

		template<typename T>
		double get_snr(const Map<VectorX<T>>& signal,
			const VectorX<T>& noise) {
			T aps = signal.squaredNorm() / static_cast<T>(signal.size());
			T apn = noise.squaredNorm() / static_cast<T>(noise.size());
			return 10 * std::log10(aps / apn);
		}
	};
}

#endif //_PPHASE_PICKER