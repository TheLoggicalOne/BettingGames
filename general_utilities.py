import time
import copy


def function_power(func, power):
    def func_power(start):
        a = start
        for i in range(power):
            a = func(a)
        return a

    return func_power


def all_time_deltas(dict_of_times):
    dict_copy = copy.deepcopy(dict_of_times)
    for key1, value1 in dict_of_times.items():
        for key2, value2 in dict_of_times.items():
            dict_copy['key1'+'to'+'key2'] = dict_of_times[key2]-dict_of_times[key1]
    return dict_copy






t = {0: time.perf_counter()}

time.sleep(5)

t[1] = time.perf_counter()
time.sleep(1)
t[2] = time.perf_counter()
