import PphasePicker
import numpy as np

'''
Тестовый запуск программы нахождения времени пикирования фазы P-волны.

Программа вызывается функцией "calculatePphasePicker" из модуля "PphasePicker".

Аргументы функции:
    Обязяательные:
        x - Входной сигнал, передается в виде numpy-массива (dtype=np.float64)
        dt = шаг дискретизации (type = float64)
    Необязательные:
        flhp - нижняя и верхняя частоты полосовой фильтрации Баттерворта, передаются в виде кортежа (flp, fhp). 
                Если передается None или ничего не передается, то фильтрация не используется
        Tn - частота осциллятора до демпфирования. По умолчанию: если dt <= 0.01, то Tn = 0.01, иначе Tn = 0.1
        xi - коэффициент демпфирования. По умолчанию значение 0.6
	jump_finder - тип алгоритма нахождения скачка у мощности демпфирования, 2 варианта: 
				PphasePicker.HIST_FINDER (по умолчанию) - метод гистограмм из оригинала, учитывается nbins;
				PphasePicker.SIGMA_FINDER - метод, основанный на правиле 3-сигм, но берется 1-сигма, nbins не учитывается.
					Примечание: на время обновления имеется гипотеза, что SIGMA_FINDER находит время пикирования S-волны, которая
					получает подтверждения на текущих тестах
        nbins - количество бинов построения гистограммы. По умолчанию: если dt <= 0.01, то int(2/dt), иначе 200
        to_peak - булево значение: если True, то входной сигнал для обработки берется до максимального значения этого сигнала по модулю
                                    иначе для обработки берется весь сигнал. Предназначен для ускорения работы программы

Выходные значения:
    (t, snr) - кортеж, где t - время пикирования (в секундах), snr - signal-noise-ratio
'''

def test_pphase_picker(file_name, flhp, Tn, xi, nbins, to_peak):
	dt = np.fromfile(file_name, dtype=np.float64, count=1)[0]
	x_len = np.fromfile(file_name, dtype=np.int32, count=1, offset=8)[0]
	x = np.fromfile(file_name, dtype=np.float64, offset=12, count=x_len)
	res = PphasePicker.calculatePphasePicker(x, dt, flhp=flhp, Tn=Tn, xi=xi, nbins=nbins, to_peak=to_peak)
	print(res)

if __name__ == '__main__':
	test_pphase_picker('strong-motion.bin', (0.1, 20), 0.01, 0.6, 400, True)
	test_pphase_picker('weak-motion.bin', None, 0.01, 0.6, 200, True)
	test_pphase_picker('acc-motion_low_sampling.bin', None, 0.1, 0.6, 200, True)
