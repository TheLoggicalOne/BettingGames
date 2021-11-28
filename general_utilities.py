def function_power(func, power):
    def func_power(start):
        a = start
        for i in range(power):
            a = func(a)
        return a
    return func_power



