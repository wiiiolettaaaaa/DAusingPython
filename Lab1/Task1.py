import numpy as np

fixed_array = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])

random_array = np.random.random((4, 3))
range_array = np.arange(1, 16, 2)
ones_array = np.ones(9, dtype=int)
zeros_array = np.zeros((2, 4), dtype=float)
linspace_array = np.linspace(0, 1, 5)
normal_array = np.random.normal(1, 10, size=(2, 5))
randint_array = np.random.randint(0, 100, (4, 4))
empty_array = np.empty(5)

print("\n--- Атрибути fixed_array ---")
print("Shape:", fixed_array.shape)
print("Dimensions:", fixed_array.ndim)
print("Size:", fixed_array.size)
print("Data type:", fixed_array.dtype)
print("Bytes:", fixed_array.nbytes)
print("Item size:", fixed_array.itemsize)

# 2. Індексація
first_element = fixed_array[0, 0]
last_element = fixed_array[-1, -1]
subarray_2d = fixed_array[0:2, 1:3]
subarray_1d = fixed_array[0, :]
subarray_1d_from_1d = range_array[2:6]
reversed_array = fixed_array[::-1, ::-1]

print("\n--- Індексація ---")
print("First element:", first_element)
print("Last element:", last_element)
print("Subarray 2D:\n", subarray_2d)
print("Subarray 1D:\n", subarray_1d)
print("Reversed array:\n", reversed_array)

# Копія підмасиву
sub_copy = fixed_array[0:2, 0:2].copy()

# 3. Арифметичні операції
add_result = fixed_array + 2
multiply_result = fixed_array * 3
divide_result = fixed_array / 2
power_result = fixed_array ** 2
mod_result = fixed_array % 2
negate_result = -fixed_array

print("\n--- Арифметичні операції ---")
print("Addition:\n", add_result)
print("Multiplication:\n", multiply_result)
print("Division:\n", divide_result)
print("Power:\n", power_result)
print("Modulo:\n", mod_result)
print("Negation:\n", negate_result)

# 4. Reduce, accumulate, outer
array_to_reduce = np.arange(0, 6)
reduce_result = np.add.reduce(array_to_reduce)
accumulate_result = np.add.accumulate(array_to_reduce)
outer_result = np.multiply.outer(np.arange(1, 10), np.arange(1, 10))

print("\n--- Reduce / Accumulate / Outer ---")
print("Reduce result:", reduce_result)
print("Accumulate result:", accumulate_result)
print("Outer result:\n", outer_result)

# 5. Статистичні характеристики
print("\n--- Статистика ---")
print("Sum:", np.sum(fixed_array))
print("Min:", np.min(fixed_array))
print("Max:", np.max(fixed_array))
print("Mean:", np.mean(fixed_array))
print("Std:", np.std(fixed_array))
print("Variance:", np.var(fixed_array))
print("Median:", np.median(fixed_array))
print("50th percentile:", np.percentile(fixed_array, 50))