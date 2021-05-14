#ifdef _DEBUG
	#undef _DEBUG
#endif

#include "common.h"

#include <tuple>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "phase_picker.h"

namespace py = pybind11;

using namespace phase_picker;
using namespace pybind11::literals;

template <typename T>
py::tuple phase_picker_py_calculate(PhasePicker &picker,
	py::array_t<T, py::array::c_style | py::array::forcecast> x,
	std::size_t nbins = 200,
	T min_edi_ratio = 4.0,
	idx_t<T> interval_after_supposed_s_pick = 200)
{

	py::buffer_info x_info = x.request();
	if (1 != x_info.ndim)
	{
		throw std::runtime_error("Error: Input signal shape must be one-dimensional");
	}

	Map<VectorX<T>> signal_map(static_cast<T*>(x_info.ptr), x_info.size);
	auto res = picker.calculate(signal_map, nbins, min_edi_ratio, interval_after_supposed_s_pick);

	return py::make_tuple(res.pphase_time_arrival, res.sphase_time_arrival, res.snr_decibel);
}

void picker_filter(PhasePicker &picker,
					py::tuple flhp,
					PICKER_TYPE picker_t)
{
	if (flhp.is_none() || flhp.size() != 2)
		return;
	double flp = flhp[0].cast<double>();
	double fhp = flhp[1].cast<double>();

	picker.set_filter(flp, fhp, picker_t);
}

PYBIND11_MODULE(P_S_PhasePicker, p_s_phase_picker_m) {
	p_s_phase_picker_m.doc() = R"pbdoc(
		Phase picker module for P and S waves
		-------------------------------------

		.. currentmodule:: P_S_PhasePicker

		.. autosummary::
			:toctree: _generate

			PhasePicker
	)pbdoc";

	py::enum_<window_function_t>(p_s_phase_picker_m, "smooth_function")
		.value("BOXCAR", window_function_t::BOXCAR)
		.value("GAUSSIAN", window_function_t::GAUSSIAN)
		.value("HAMMING", window_function_t::HAMMING)
		.value("HANN", window_function_t::HANN)
		.value("HANNING", window_function_t::HANNING)
		.value("KONNO_OHMACHI", window_function_t::KONNO_OHMACHI)
		.value("PARZEN", window_function_t::PARZEN)
		.value("TRIANG", window_function_t::TRIANG)
		.value("DEFAULT", window_function_t::DEFAULT)
		.export_values();

	py::class_<PhasePicker>(p_s_phase_picker_m, "PhasePicker")
		.def(py::init([] (double dt)
		{
			std::unique_ptr<PhasePicker> phase_picker_p{new PhasePicker(dt)};
			phase_picker_p->set_filter(2.0, 30.0, PICKER_TYPE::S_PICKER);
			phase_picker_p->set_smooth_function(window_function_t::KONNO_OHMACHI, smooth_function_t::BASE_WINDOW_SIZE, PICKER_TYPE::S_PICKER);
			phase_picker_p->set_smooth_function(window_function_t::HANN, smooth_function_t::BASE_WINDOW_SIZE, PICKER_TYPE::P_PICKER);
			phase_picker_p->set_Tn(0.01, PICKER_TYPE::P_PICKER);
			phase_picker_p->set_Tn(0.5, PICKER_TYPE::S_PICKER);
			phase_picker_p->set_to_peak(false, PICKER_TYPE::P_PICKER);
			phase_picker_p->set_to_peak(false, PICKER_TYPE::S_PICKER);
			phase_picker_p->set_damping_ratio(0.6, PICKER_TYPE::P_PICKER);
			phase_picker_p->set_damping_ratio(0.6, PICKER_TYPE::S_PICKER);
			
			return phase_picker_p;
		}), py::call_guard<py::gil_scoped_release>(), R"pbdoc(
			Create PhasePicker object for further picks calculating. Signal's preprocessing parameters set by default

			Parameters
			----------
			dt : float
				Discrete step
		)pbdoc")
	
		.def("set_p_picker_filter", [] (PhasePicker &picker, py::tuple flhp) { picker_filter(picker, flhp, PICKER_TYPE::P_PICKER); }, 
			py::call_guard<py::gil_scoped_release>(), R"pbdoc(
			Enable Butterworth bandpass filter for P-picker's signal preprocessing with bandwidth (flp, fhp)

			Parameters
			----------
			flhp : (float, float), default: None
				tuple as (flp, fhp)
		)pbdoc")

		.def("set_s_picker_filter", [] (PhasePicker &picker, py::tuple flhp) { picker_filter(picker, flhp, PICKER_TYPE::S_PICKER); },
			py::call_guard<py::gil_scoped_release>(), R"pbdoc(
			Enable Butterworth bandpass filter for S-picker's signal preprocessing with bandwidth (flp, fhp)
	
			Parameters
			----------
			flhp : (float, float), default: (2.0, 30.0)
				Tuple as (flp, fhp)
		)pbdoc")

		.def("disable_p_picker_filter", [] (PhasePicker &picker) { picker.disable_filter(PICKER_TYPE::P_PICKER); }, R"pbdoc(
			Disable Butterworth bandpass filter for P-picker's signal preprocessing
		)pbdoc")

		.def("disable_s_picker_filter", [] (PhasePicker &picker) { picker.disable_filter(PICKER_TYPE::S_PICKER); }, R"pbdoc(
			Disable Butterworth bandpass filter for S-picker's signal preprocessing
		)pbdoc")

		.def("set_p_picker_Tn", [] (PhasePicker &picker, double Tn) { picker.set_Tn(Tn, PICKER_TYPE::P_PICKER); }, R"pbdoc(
			Set period for building SDOF system P-picker's signal preprocessing

			Parameters
			----------
			Tn : float, default: 0.01
				SDOF system period
		)pbdoc")

		.def("set_s_picker_Tn", [] (PhasePicker &picker, double Tn) { picker.set_Tn(Tn, PICKER_TYPE::S_PICKER); }, R"pbdoc(
			Set period for building SDOF system S-picker's signal preprocessing

			Parameters
			----------
			Tn : float, default: 0.5
				SDOF system period
		)pbdoc")

		.def("set_p_picker_damping_ratio", [] (PhasePicker &picker, double damping_ratio) { picker.set_damping_ratio(damping_ratio, PICKER_TYPE::P_PICKER); }, R"pbdoc(
			Set damping ratio for building SDOF system P-picker's signal preprocessing

			Parameters
			----------
			damping_ratio : float, default: 0.6
				SDOF system damping ratio
		)pbdoc")

		.def("set_s_picker_damping_ratio", [] (PhasePicker &picker, double damping_ratio) { picker.set_damping_ratio(damping_ratio, PICKER_TYPE::S_PICKER); }, R"pbdoc(
			Set damping ratio for building SDOF system S-picker's signal preprocessing

			Parameters
			----------
			damping_ratio : float, default: 0.6
				SDOF system damping ratio
		)pbdoc")

		.def("set_p_picker_to_peak", [] (PhasePicker &picker, bool to_peak) { picker.set_to_peak(to_peak, PICKER_TYPE::P_PICKER); }, R"pbdoc(
			Enable or disable P-picker's signal processing until abs peak

			Parameters
			----------
			to_peak : bool, default: False
				Enable or disable to peak processing
		)pbdoc")

		.def("set_s_picker_to_peak", [] (PhasePicker &picker, bool to_peak) { picker.set_to_peak(to_peak, PICKER_TYPE::S_PICKER); }, R"pbdoc(
			Enable or disable S-picker's signal processing until abs peak

			Parameters
			----------
			to_peak : bool, default: False
				Enable or disable to peak processing
		)pbdoc")

		.def("set_p_picker_smooth_function", [] (PhasePicker &picker, window_function_t window_f, unsigned long window_size = smooth_function_t::BASE_WINDOW_SIZE) 
			{ picker.set_smooth_function(window_f, window_size, PICKER_TYPE::P_PICKER); }, "window_f"_a, "window_size"_a=smooth_function_t::BASE_WINDOW_SIZE, R"pbdoc(
			Set window smooting function for SDOF damping power (P-picker)

			Parameters
			----------
			window_f : smooth_function, default: smooth_function.HANN
				Smoothing function type
			window_size : int, default: 40
				Smoothing window
		)pbdoc")

		.def("set_s_picker_smooth_function", [] (PhasePicker &picker, window_function_t window_f, unsigned long window_size = smooth_function_t::BASE_WINDOW_SIZE) 
			{ picker.set_smooth_function(window_f, window_size, PICKER_TYPE::S_PICKER); }, "window_f"_a, "window_size"_a=smooth_function_t::BASE_WINDOW_SIZE, R"pbdoc(
			Set window smooting function for SDOF damping power (S-picker)

			Parameters
			----------
			window_f : smooth_function, default: smooth_function.KONNO_OHMACHI
				Smoothing function type
			window_size : int, default: 40
				Smoothing window
		)pbdoc")
	
		.def("set_dt", &PhasePicker::set_dt, R"pbdoc(
			Set dicrete step for signal

			Parameters
			----------
			dt : float
				Discrete step
		)pbdoc")

		.def("get_dt", &PhasePicker::get_dt, R"pbdoc(
			Get dicrete step for signal

			Returns
			-------
			float
				Discrete step
		)pbdoc")
	
		.def("calculate", phase_picker_py_calculate<double>,
			"x"_a, "nbins"_a=200, "min_edi_ratio"_a=4.0, "interval_after_supposed_s_pick"_a=200,  
			py::call_guard<py::gil_scoped_release>(), R"pbdoc(
			Calculating P and S-phase picks for signal

			Parameters
			----------
			x : array_like (1 dimension)
				Signal for processing
			min_edi_ratio : float, default: 4.0
				Min energy ratio before and after supposed detecting S-wave
			interval_after_supposed_s_pick : int, default: 200
				Number of samples after supposed detecting S-wave for calculating energy ratio
			
			Returns
			-------
			(float, float, float)
				Tuple with S-pick time, P-pick time and SNR in dB
		)pbdoc")
	
		.def("calculate", phase_picker_py_calculate<float>,
			"x"_a, "nbins"_a=200, "min_edi_ratio"_a=4.0, "interval_after_supposed_s_pick"_a=200,  
			py::call_guard<py::gil_scoped_release>(), R"pbdoc(
			Calculating P and S-phase picks for signal

			Parameters
			----------
			x : array_like (1 dimension)
				Signal for processing
			min_edi_ratio : float, default: 4.0
				Min energy ratio before and after supposed detecting S-wave
			interval_after_supposed_s_pick : int, default: 200
				Number of samples after supposed detecting S-wave for calculating energy ratio
			
			Returns
			-------
			(float, float, float)
				Tuple with S-pick time, P-pick time and SNR in dB
		)pbdoc");
}