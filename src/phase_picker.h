#ifndef _PHASE_PICKER
#define _PHASE_PICKER

#include "common.h"

#include <cstddef>
#include <cmath>
#include <algorithm>
#include <limits>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include "butterworth_bandpass.h"
#include "filter.h"
#include "math_utils.h"

namespace phase_picker {

    struct pphase_picker_result {
        double pphase_time_arrival = -1.0;
        double snr_decibel = -1.0;
    };

    struct p_s_phase_picker_result {
        double pphase_time_arrival = -1.0;
        double sphase_time_arrival = -1.0;
        double snr_decibel = -1.0;
    };

    struct smooth_function_t {
        static constexpr unsigned long BASE_WINDOW_SIZE = 40;
        window_function_t function_type = window_function_t::DEFAULT;
        unsigned long window_size = BASE_WINDOW_SIZE;
    };

    struct picker_filter_t {
        double flp = -1.0;
        double fhp = -1.0;
        bool is_filtering = false;
        FiltFiltButter<4> filter;
    };

    struct signal_preprocessing_t {
        double Tn = 0.01;
        double damping_ratio = 0.6;
        bool to_peak = false;
        picker_filter_t picker_filter = {-1.0, -1.0, false};
        smooth_function_t picker_smooth_function = {window_function_t::DEFAULT, smooth_function_t::BASE_WINDOW_SIZE};
    };

    enum class PICKER_TYPE {
        P_PICKER,
        S_PICKER
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

    class PhasePicker {
    public:
        explicit PhasePicker(double dt) :
                dt_(dt)
        { }

        void set_filter(double picker_flp,
                        double picker_fhp,
                        PICKER_TYPE picker_t)
        {
            if (picker_t == PICKER_TYPE::P_PICKER) {
                set_filter(picker_flp, picker_fhp, p_picker_preprocessing_.picker_filter);
            } else if (picker_t == PICKER_TYPE::S_PICKER) {
                set_filter(picker_flp, picker_fhp, s_picker_preprocessing_.picker_filter);
            }
        }

        void disable_filter(PICKER_TYPE picker_t) 
        {
            if (picker_t == PICKER_TYPE::P_PICKER) 
            {
                p_picker_preprocessing_.picker_filter.is_filtering = false;
            } else if (picker_t == PICKER_TYPE::S_PICKER) 
            {
                s_picker_preprocessing_.picker_filter.is_filtering = false;
            }
        }

        void set_smooth_function(window_function_t window_f,
                                unsigned long window_size,
                                PICKER_TYPE picker_t) {
            if (picker_t == PICKER_TYPE::P_PICKER) {
                set_smooth_function(window_f, window_size, p_picker_preprocessing_.picker_smooth_function);
            }
            if (picker_t == PICKER_TYPE::S_PICKER) {
                set_smooth_function(window_f, window_size, s_picker_preprocessing_.picker_smooth_function);
            }
        }

        void set_dt(double dt)
        {
            dt_ = dt;
            create_filter(p_picker_preprocessing_.picker_filter);
            create_filter(s_picker_preprocessing_.picker_filter);
        }

        void set_Tn(double Tn,
                    PICKER_TYPE picker_t)
        {
            if (picker_t == PICKER_TYPE::P_PICKER)
            {
                p_picker_preprocessing_.Tn = Tn;
            } else if (picker_t == PICKER_TYPE::S_PICKER)
            {
                s_picker_preprocessing_.Tn = Tn;
            }
        }

        void set_damping_ratio(double damping_ratio,
                                PICKER_TYPE picker_t)
        {
            if (picker_t == PICKER_TYPE::P_PICKER)
            {
                p_picker_preprocessing_.damping_ratio = damping_ratio;
            } else if (picker_t == PICKER_TYPE::S_PICKER)
            {
                s_picker_preprocessing_.damping_ratio = damping_ratio;
            }
        }

        void set_to_peak(bool to_peak,
                        PICKER_TYPE picker_t)
        {
            if (picker_t == PICKER_TYPE::P_PICKER)
            {
                p_picker_preprocessing_.to_peak = to_peak;
            } else if (picker_t == PICKER_TYPE::S_PICKER)
            {
                s_picker_preprocessing_.to_peak = to_peak;
            }
        }

        double get_dt() const
        {
            return dt_;
        }

        template<typename T>
        p_s_phase_picker_result calculate(
            const VectorX<T> &x, \
            std::size_t nbins, \
            double min_edi_ratio, \
            std::size_t interval_after_supposed_s_pick, \
            std::size_t skipped_nbins) {
            VectorX<T> signal_for_p_picker, signal_for_s_picker;
            std::tie(signal_for_p_picker, signal_for_s_picker) = prepare_signal(x);

            VectorX<T> p_picker_velocity = construct_sdof_system(signal_for_p_picker, p_picker_preprocessing_.Tn, p_picker_preprocessing_.damping_ratio).row(1).transpose();
            VectorX<T> s_picker_velocity = construct_sdof_system(signal_for_s_picker, s_picker_preprocessing_.Tn, s_picker_preprocessing_.damping_ratio).row(1).transpose();

            auto p_picker_result = pphase_picker_calculate_impl(signal_for_p_picker, p_picker_velocity, nbins, x);

            auto result = sphase_picker_calculate_impl(signal_for_s_picker, \
                s_picker_velocity, \
                nbins, \
                min_edi_ratio, \
                interval_after_supposed_s_pick, \
                skipped_nbins, \
                p_picker_result);

            return result;
        }
    private:

        template <typename T>
        p_s_phase_picker_result sphase_picker_calculate_impl(const VectorX<T> &signal, \
            const VectorX<T>& velocity, \
            std::size_t nbins, \
            double min_edi_ratio, \
            std::size_t interval_after_supposed_s_pick, \
            std::size_t skipped_nbins, \
            pphase_picker_result p_picker_result) {
            p_s_phase_picker_result result;

            double Tn = s_picker_preprocessing_.Tn;
            double damping_ratio = s_picker_preprocessing_.damping_ratio;
            unsigned long window_size = s_picker_preprocessing_.picker_smooth_function.window_size;
            window_function_t window_f = s_picker_preprocessing_.picker_smooth_function.function_type;

            const T omega_n = 2.0 * T(M_PI) / T(Tn);
            const T C = 2.0 * T(damping_ratio) * omega_n;

            if (std::abs(p_picker_result.pphase_time_arrival + 1.0) < std::numeric_limits<double>::epsilon()) {
                return {-1.0, -1.0, -1.0};
            }

            auto pp_loc = static_cast<idx_t<T>>(p_picker_result.pphase_time_arrival / dt_);
            double s_pick = -1;

            VectorX<T> d_energy_damping = integrand_energy_damping(velocity, C, window_size, window_f);

            idx_t<T> max_d_en_i;
            auto max_d_en = d_energy_damping.maxCoeff(&max_d_en_i);

            VectorX<T> energy_damping = math_utils::integrate(d_energy_damping, dt_);

            const VectorX<decltype(nbins)> nbins_seq = VectorX<decltype(nbins)>::LinSpaced((nbins - 5) / 5, 5, nbins);

            VectorX<T> ratio = VectorX<T>::Zero(nbins_seq.size());

            VectorX<idx_t<T>> dloc = VectorX<idx_t<T>>::Zero(nbins_seq.size());

            VectorX<idx_t<T>> sp_loc_all = VectorX<idx_t<T>>::Zero(nbins_seq.size());
            for (auto i_bin = 0; i_bin < nbins_seq.size(); ++i_bin) {
                sp_loc_all[i_bin] = get_lock_hist(d_energy_damping, \
                                                nbins_seq[i_bin], \
                                                PICKER_TYPE::S_PICKER, \
                                                skipped_nbins);
                std::tie(ratio[i_bin], dloc[i_bin]) = find_kink(energy_damping, \
                    d_energy_damping, \
                    std::make_pair(pp_loc, sp_loc_all[i_bin]), \
                    interval_after_supposed_s_pick, \
                    min_edi_ratio);
            }

            idx_t<T> max_ratio_i;

            auto max_ratio = ratio.maxCoeff(&max_ratio_i);

            if (max_ratio > T(0.0)) {
                s_pick = sp_loc_all[max_ratio_i] * dt_;
            }


            result.pphase_time_arrival = p_picker_result.pphase_time_arrival;
            result.sphase_time_arrival = s_pick;
            result.snr_decibel = p_picker_result.snr_decibel;

            return result;
        }

        template <typename T>
        pphase_picker_result pphase_picker_calculate_impl(const VectorX<T>& signal, \
            const VectorX<T> &velocity, \
            std::size_t nbins, \
            const VectorX<T> &x) {
            pphase_picker_result result;

            double Tn = p_picker_preprocessing_.Tn;
            double damping_ratio = p_picker_preprocessing_.damping_ratio;
            unsigned long window_size = p_picker_preprocessing_.picker_smooth_function.window_size;
            window_function_t window_f = p_picker_preprocessing_.picker_smooth_function.function_type;

            const T omega_n = 2.0 * T(M_PI) / T(Tn);
            const T C = 2.0 * T(damping_ratio) * omega_n;

            auto d_energy_damping = integrand_energy_damping(velocity, C, window_size, window_f);

            double loc = -1.0;
            idx_t<T> pick = -1;

            pick = get_pick_hist(d_energy_damping, signal, nbins);

            if (pick == -1) {
                pick = get_pick_hist(d_energy_damping, \
                    signal, \
                    std::ceil(static_cast<double>(nbins) / 2.0));
                if (pick != -1) {
                    loc = (static_cast<double>(pick) + 1.0) * dt_;
                }
            } else {
                loc = (static_cast<double>(pick) + 1.0) * dt_;
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

        template <typename T>
        auto prepare_signal(const VectorX<T> &x) -> std::pair<VectorX<T>, VectorX<T>> {
            VectorX<T> signal = x;
            T inf_norm_inverse = static_cast<T>(1.0) / signal.template lpNorm<Infinity>();
            signal *= inf_norm_inverse;
            math_utils::detrend(signal, signal);

            VectorX<T> signal_for_s_picker, signal_for_p_picker;
            if (s_picker_preprocessing_.picker_filter.is_filtering) {
                signal_for_s_picker = s_picker_preprocessing_.picker_filter.filter.template filtfilt<T>(signal);
            } else {
                signal_for_s_picker = signal;
            }

            if (p_picker_preprocessing_.picker_filter.is_filtering) {
                signal_for_p_picker = p_picker_preprocessing_.picker_filter.filter.template filtfilt<T>(signal);
            } else {
                signal_for_p_picker = signal;
            }

            if (p_picker_preprocessing_.to_peak) {
                idx_t<T> end_signal;
                T max_val = signal_for_p_picker.cwiseAbs().maxCoeff(&end_signal);
                signal_for_p_picker = signal_for_p_picker.head(end_signal + 1);
            }
            if (s_picker_preprocessing_.to_peak) {
                idx_t<T> end_signal;
                T max_val = signal_for_s_picker.cwiseAbs().maxCoeff(&end_signal);
                signal_for_s_picker = signal_for_s_picker.head(end_signal + 1);
            }

            return std::make_pair(signal_for_p_picker, signal_for_s_picker);
        }
            
        template <typename T>
        Matrix<T, Dynamic, Dynamic>
        construct_sdof_system(const VectorX<T> & signal, \
            double Tn, \
            double damping_ratio) {
            const T omega_n = 2.0 * T(M_PI) / T(Tn);
            const T C = 2.0 * T(damping_ratio) * omega_n;
            const T K = std::pow(omega_n, 2);
            auto end_signal = signal.size();

            Matrix<T, 2, 2> A;
            A << 0.0, 1.0,
                -K, -C;

            static const Matrix<T, 2, 2> eye_2x2 = Matrix<T, 2, 2>::Identity();
            static const Matrix<T, 2, 1> zero_one = (Matrix<T, 2, 1>() << 0, 1).finished();

            Matrix<T, Dynamic, Dynamic> y(2, end_signal);
            y.col(0) << 0.0, 0.0;
            Matrix<T, 2, 2> Ae = (A * dt_).exp().template cast<T>();
            Matrix<T, 2, 1> AeB = (A.inverse() * (Ae - eye_2x2)) * zero_one;

            for (decltype(end_signal) i = 1; i < end_signal; ++i) {
                y.col(i) = (Ae * y.col(i - 1)) + (AeB * signal[i]);
            }

            return y;
        }

        template <typename T>
        VectorX<T> integrand_energy_damping(const VectorX<T> & velocity, \
            T C, \
            unsigned long window_size, \
            window_function_t window_f) {
            return math_utils::smooth_curve(static_cast<VectorX<T>>(C * velocity.cwiseAbs2()), window_size, window_f);
        }

        template <typename T>
        auto find_kink(const VectorX<T> &energy_damping, \
            const VectorX<T> &d_energy_damping, \
            const std::pair<idx_t<T>, idx_t<T>> &p_s_locks, \
            idx_t<T> interval_after_supposed_s_pick, \
            double min_edi_ratio) -> std::pair<T, idx_t<T>> {
            using to_T = T;
            constexpr unsigned long order = 4;

            T ratio = -1;
            idx_t<T> dloc = -1;

            idx_t<T> p_loc, s_loc;
            std::tie(p_loc, s_loc) = p_s_locks;

            if (p_loc > s_loc) 
            {
                return std::make_pair(ratio, dloc);
            }

            VectorX<T> before_s_candidate_samples = VectorX<T>::LinSpaced(s_loc - p_loc + 1, to_T(p_loc), to_T(s_loc));
            VectorX<T> before_s_candidate_signals = energy_damping.segment(p_loc, s_loc - p_loc + 1);

            if (interval_after_supposed_s_pick + s_loc >= energy_damping.size()) {
                interval_after_supposed_s_pick = energy_damping.size() - s_loc - 1;
            }

            auto peak_before_s_candidate = d_energy_damping.segment(p_loc, s_loc - p_loc + 1).maxCoeff();

            VectorX<T> after_s_candidate_signals = d_energy_damping.segment(s_loc, interval_after_supposed_s_pick);
            auto peaks_after_s_candidate = math_utils::find_local_maximums(after_s_candidate_signals);

            auto max_peak_after_s_candidate = std::max_element(std::cbegin(peaks_after_s_candidate), 
                                                            std::cend(peaks_after_s_candidate), 
                [] (const std::pair<T, idx_t<T>> &v1, const std::pair<T, idx_t<T>> &v2) {
                    return v1.first < v2.first;
                })->first;

            if (max_peak_after_s_candidate > min_edi_ratio*peak_before_s_candidate) {
                dloc = peaks_after_s_candidate[0].second;

                VectorX<T> after_s_candidate_samples = VectorX<T>::LinSpaced(dloc + 1, to_T(s_loc), to_T(s_loc + dloc));
                after_s_candidate_signals = energy_damping.segment(s_loc, dloc + 1);

                ratio = math_utils::vector_gradient(math_utils::polyfit(after_s_candidate_samples, after_s_candidate_signals, order)).lpNorm<1>()
                    / math_utils::vector_gradient(math_utils::polyfit(before_s_candidate_samples, before_s_candidate_signals, order)).lpNorm<1>();
            }



            return std::make_pair(ratio, dloc);
        }

        template<typename T>
        idx_t<T> get_pick_hist(const VectorX<T>& d_energy_damping,
            const VectorX<T>& signal,
            std::size_t bin_size) {

            auto loc_i = get_lock_hist(d_energy_damping, bin_size, PICKER_TYPE::P_PICKER);
            idx_t<T> pick = -1;
            if (loc_i == -1) 
            {
                return pick;
            }
            
            VectorX<T> signal_cross = signal.segment(0, loc_i).cwiseProduct(signal.segment(1, loc_i));
            for (idx_t<T> i = signal_cross.size(); i > 0; --i) {
                if (signal_cross[i-1] < 0) {
                    pick = i-1;
                    break;
                }
            }

            return pick;
        }

        template <typename T>
        idx_t<T> get_lock_hist(const VectorX<T>& d_energy_damping, \
            std::size_t bin_size, \
            PICKER_TYPE picker_t, \
            std::size_t skipped_nbins = 0) {
            
            Vector2d R = statelevel(d_energy_damping, bin_size, picker_t, skipped_nbins);
            
            auto R_0 = R[0];

            idx_t<T> loc_i = -1;
            for (idx_t<T> i = 0; i < d_energy_damping.size(); ++i) {
                if (d_energy_damping[i] > R_0)
                {
                    loc_i = i;
                    break;
                }
            }

            return loc_i;
        }

        template<typename T>
        Vector2d statelevel(const VectorX<T>& y, \
            const std::size_t nbins, \
            PICKER_TYPE picker_t, \
            std::size_t skipped_nbins) {

            auto histogram = math_utils::function_hisotgram(y, nbins);

            auto y_min = y.minCoeff();
            auto y_max = y.maxCoeff();
            auto Ry = y_max - y_min;
            auto dy = Ry / nbins;

            VectorXi::Index i_low = -1;
            for (auto i = 0; i < histogram.size(); ++i) {
                if (histogram[i] > 0) {
                    i_low = i;
                    break;
                }
            }
            VectorXi::Index i_high = -1;
            for (auto i = histogram.size(); i > 0; --i) {
                if (histogram[i-1] > 0) {
                    i_high = i-1;
                    break;
                }
            }

            const auto l_low = i_low;
            const VectorXi::Index l_high = i_low + std::floor((static_cast<double>(i_high) - static_cast<double>(i_low)) / 2.0);
            const auto u_low = l_high;
            const auto u_high = i_high;

            const VectorXi l_hist = histogram.segment(l_low, l_high - l_low + 1);
            const VectorXi u_hist = histogram.segment(u_low, u_high - u_low + 1);

            Vector2d levels = Vector2d::Zero();

            typename VectorXi::Index i_max_l_hist, i_max_u_hist;
            if (picker_t == PICKER_TYPE::P_PICKER) {
                l_hist.maxCoeff(&i_max_l_hist);
                u_hist.maxCoeff(&i_max_u_hist);
            } else if (picker_t == PICKER_TYPE::S_PICKER) {
                l_hist.tail(skipped_nbins).maxCoeff(&i_max_l_hist);
                i_max_l_hist += skipped_nbins;
                u_hist.tail(skipped_nbins).maxCoeff(&i_max_u_hist);
                i_max_u_hist += skipped_nbins;
            }

            levels[0] = y_min + dy * (static_cast<double>(l_low) + static_cast<double>(i_max_l_hist) + 0.5);
            levels[1] = y_min + dy * (static_cast<double>(u_low) + static_cast<double>(i_max_u_hist) + 0.5);

            return levels;
        }

        template<typename T>
        double get_snr(const VectorX<T> &signal,
            const VectorX<T>& noise) {
            T aps = signal.squaredNorm() / static_cast<T>(signal.size());
            T apn = noise.squaredNorm() / static_cast<T>(noise.size());
            return 10. * std::log10(aps / apn);
        }

        void set_smooth_function(window_function_t window_f,
                                unsigned long window_size,
                                smooth_function_t &smooth_f)
        {
            smooth_f.function_type = window_f;
            smooth_f.window_size = window_size;
        }

        void set_filter(double flp,
                        double fhp,
                        picker_filter_t &picker_filter)
        {
            picker_filter.flp = flp;
            picker_filter.fhp = fhp;
            picker_filter.is_filtering = true;
            create_filter(picker_filter);
        }

        void create_filter(picker_filter_t &picker_filter) 
        {
            if (picker_filter.is_filtering) 
            {
                picker_filter.filter = FiltFiltButter<4>::create_butterworth_bandpass_filter(dt_, picker_filter.flp, picker_filter.fhp);	
            }
            
        }

        private:
            double dt_;
            signal_preprocessing_t p_picker_preprocessing_;
            signal_preprocessing_t s_picker_preprocessing_;
    };
}

#endif //_PHASE_PICKER