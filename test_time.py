import timeit

# Definindo os números e a potência
base = 45
exponent = 20

# Método 1: Usando a função pow()
def method_pow():
    return pow(base, exponent)

# Método 2: Usando o operador de exponenciação **
def method_operator():
    return base ** exponent

# Método 3: Usando a função pow() de NumPy
def method_numpy_pow():
    import numpy as np
    return np.power(base, exponent)

# Método 4: Usando o operador de exponenciação ** com NumPy
def method_numpy_operator():
    import numpy as np
    return base ** exponent

# Testando e comparando os métodos de exponenciação
print("Usando a função pow():", timeit.timeit(method_pow, number=1000000))
print("Usando o operador **:", timeit.timeit(method_operator, number=1000000))
print("Usando numpy.power():", timeit.timeit(method_numpy_pow, number=1000000))
print("Usando operador ** com NumPy:", timeit.timeit(method_numpy_operator, number=1000000))
