import numpy as np

nan = np.nan
arr_1 = np.array([ 1., nan, nan, nan, nan, nan])
arr_2 = np.array([ nan, nan, nan, nan, nan, nan])

list_of_arrays = [arr_1,arr_2]
filtered_list = [arr for arr in list_of_arrays if not np.all(np.isnan(arr))]

o = np.nanmean(filtered_list, axis=0)
print(o)