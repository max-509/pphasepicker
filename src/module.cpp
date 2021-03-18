#ifdef _DEBUG
	#undef _DEBUG
#endif
#include "pphase_picker.h"

#include <iostream>
#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>

using namespace pphase_picker;

static PyObject *CompError;

static PyObject *
pphase_picker_calculate(PyObject *self, PyObject *args, PyObject *kwargs) {
	static const char* kwargs_names[] = {"x",
										"dt",
										"flhp",
										"Tn",
										"xi",
										"nbins",
										"to_peak",
										nullptr};

	PyObject *signal_o = nullptr;
	double dt;
	Py_XINCREF(Py_None);
	PyObject *flhp_o = Py_None;
	double Tn = -1;
	double damping_ratio = 0.6;
	unsigned long nbins = 0;
	int to_peak_i = 1;
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Od|$Oddkp", const_cast<char**>(kwargs_names), &signal_o, &dt, &flhp_o, &Tn, &damping_ratio, &nbins, &to_peak_i)) {
		return nullptr;
	}

	bool is_filter = false;
	double flp = 0.0, fhp = 0.0;

	if (flhp_o != Py_None) {
		if (PyTuple_Check(flhp_o) == 1 && PyTuple_Size(flhp_o) == 2) {
			if (!PyArg_ParseTuple(flhp_o, "dd", &flp, &fhp)) {
				return nullptr;
			}
			is_filter = true;
		}
	}

	if (Tn <= 0) {
		if (dt <= 0.01) {
			Tn = 0.01;
		} else {
			Tn = 0.1;
		}
	}

	if (nbins == 0) {
		if (dt <= 0.01) {
			nbins = static_cast<unsigned long>(2.0 / dt);
		} else {
			nbins = 200;
		}
	}

	bool to_peak = true;
	if (to_peak_i == 0) {
		to_peak = false;
	} else {
		to_peak = true;
	}

	signal_o = PyArray_FROM_OTF(signal_o, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
	if (signal_o == nullptr) {
		return nullptr;
	}
	const npy_intp signal_len = PyArray_DIMS(signal_o)[0];
	double *signal_arr = (double *)PyArray_DATA(signal_o);
	Map<VectorXd> signal_map(signal_arr, signal_len);

	PphasePicker pphase_picker_calculator(dt, is_filter, flp, fhp);
	pphase_picker_result res = pphase_picker_calculator.calculate<double>(signal_map,
																Tn,
																damping_ratio,
																nbins,
																to_peak);

	Py_XDECREF(signal_o);

	return Py_BuildValue("dd", (res.pphase_time_arrival), (res.snr_decibel));

}

static PyMethodDef methods[] =
{
	{"calculatePphasePicker", (PyCFunction)(void(*)(void))pphase_picker_calculate, METH_VARARGS | METH_KEYWORDS, "Find onset time of the P-wave phase"},
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "pphase_picker",
    "Pphase picker package for detection begin of event",
    -1,
    methods
};

PyMODINIT_FUNC
PyInit_PphasePicker() {

    import_array();

    PyObject* m = nullptr;
    m = PyModule_Create(&module);
    if (m == nullptr) return nullptr;
    CompError = PyErr_NewException("compute.error", nullptr, nullptr);
    Py_INCREF(CompError);
    PyModule_AddObject(m, "error", CompError);

    return m;
}