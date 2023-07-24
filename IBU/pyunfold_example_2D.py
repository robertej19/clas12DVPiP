import numpy as np
from pyunfold import iterative_unfold
from scipy.stats import norm
from pyunfold.priors import uniform_prior
from pyunfold.priors import jeffreys_prior
import pandas as pd
from utils import filestruct, const, make_histos
import numpy as np
import sys, os

from pyunfold import iterative_unfold
from pyunfold.callbacks import Logger

import numpy as np
np.random.seed(2)
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib as mpl
PhysicsConstants = const.PhysicsConstants()


fs = filestruct.fs()

bins_Q2,bins_xB, bins_t1, bins_phi1 = fs.Q2bins, fs.xBbins, fs.tbins, fs.phibins

uniform_prior = uniform_prior(num_causes=4)
causes = np.arange(4)
jeffreys_prior = jeffreys_prior(causes=causes)

test_file = "/mnt/d/GLOBUS/CLAS12/Thesis/2_selected_dvpip_events/inb/rec/dvpip_events_norad_10000_20230703_1814_Fall_2018_Inbending_50nA_recon.pkl"
binned_test_file = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/rec/singles_t2/binned_dvpip_events_norad_10000_20230703_1814_Fall_2018_Inbending_50nA_recon.pkl"
df = pd.read_pickle(test_file)#.head(5)
print(df.columns.values)
#remove all rows where Gent1 is larger than 1.8

cutoff = 1.7
cutoff_low = 0.1
df = df[df['Gent1'] < cutoff]
df = df[df['t1']<cutoff]
df = df[df['t1']>cutoff_low]
df = df[df['Gent1']>cutoff_low]

sns.set_context(context='poster')
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['lines.markeredgewidth'] = 2

# for 'xB' and 'GenxB'
# for 'xB' and 'GenxB'
bins_xB_extended = np.concatenate(([-10], bins_xB, [30]))

#print(bins_xB_extended)
#print(bins_xB_extended[:-1])
df['bin_xB'] = pd.cut(df['xB'], bins=bins_xB_extended, right=False, labels=bins_xB_extended[:-1])
df['bin_GenxB'] = pd.cut(df['GenxB'], bins=bins_xB_extended, right=False, labels=bins_xB_extended[:-1])

# for 'Q2' and 'GenQ2'
bins_Q2_extended = np.concatenate(([-30], bins_Q2, [30]))
df['bin_Q2'] = pd.cut(df['Q2'], bins=bins_Q2_extended, right=False, labels=bins_Q2_extended[:-1])
df['bin_GenQ2'] = pd.cut(df['GenQ2'], bins=bins_Q2_extended, right=False, labels=bins_Q2_extended[:-1])

# for 't1' and 'Gent1'
df['bin_t1'] = pd.cut(df['t1'], bins=bins_t1, right=False, labels=bins_t1[:-1])
df['bin_Gent1'] = pd.cut(df['Gent1'], bins=bins_t1, right=False, labels=bins_t1[:-1])

# for 'phi1' and 'Genphi1'
df['bin_phi1'] = pd.cut(df['phi1'], bins=bins_phi1, right=False, labels=bins_phi1[:-1])
df['bin_Genphi1'] = pd.cut(df['Genphi1'], bins=bins_phi1, right=False, labels=bins_phi1[:-1])


# create a column with unique bin combination for 'xB', 'Q2', 't1', 'phi1'
df['bin_comb'] = df['bin_t1'].astype(str) + '_' + df['bin_phi1'].astype(str)

# create a column with unique bin combination for 'GenxB', 'GenQ2', 'Gent1', 'Genphi1'
df['bin_comb_Gen'] =  df['bin_Gent1'].astype(str) + '_' + df['bin_Genphi1'].astype(str)

# create a column for unique bin numbers, using pandas' factorize method
df['bin_number'] = pd.factorize(df['bin_comb'])[0]

# similarly for the 'Gen' bin combinations
df['bin_number_Gen'] = pd.factorize(df['bin_comb_Gen'])[0]
#remove any rows where bin_t1 or bin_Gent1 is -30 or bins_t1[:-1]
# df = df[df['bin_t1'] > -30]
# df = df[df['bin_Gent1'] > -30]
# df = df[df['bin_t1'] < bins_t1[-1]]
# df = df[df['bin_Gent1'] < bins_t1[-1]]

#get the unique bins in bin_number
unique_bins = df['bin_number'].unique()

bins = bins_phi1
num_bins = len(bins) - 1

value_counts_1 = df['bin_Genphi1'].value_counts()
print(value_counts_1)

counts_series = pd.Series(0, index=bins_phi1[:-1])
counts_1 = counts_series.add(value_counts_1, fill_value=0)
data_true = counts_1.values

value_counts_2= df['bin_phi1'].value_counts()
counts_series = pd.Series(0, index=bins_phi1[:-1])
counts_2 = counts_series.add(value_counts_2, fill_value=0)
data_observed = counts_2.values





#bins_phi1 = bins_phi1.astype(float)

# Creating a zero-filled DataFrame with desired index and columns
hist_2d = pd.DataFrame(0, index=bins_phi1[:-1], columns=bins_phi1[:-1])

# Calculating the counts using crosstab
counts = pd.crosstab(df['bin_phi1'], df['bin_Genphi1'])

# Filling the actual counts into our zero-filled DataFrame
hist_2d.update(counts)

# Converting the DataFrame to numpy array
hist_2d_array = hist_2d.values

print(hist_2d_array)

# categories = pd.Categorical(bins_phi1[:-1], ordered=True)

# # Convert the 'bin_t1' and 'bin_Gent1' columns to categorical type and assign the defined categories
# df['bin_phi1'] = pd.Categorical(df['bin_phi1'], categories=categories, ordered=True)
# df['bin_Genphi1'] = pd.Categorical(df['bin_Genphi1'], categories=categories, ordered=True)

# print(df['bin_phi1'].head(10))


# # Use crosstab to compute a 2D histogram
# hist_2d = pd.crosstab(df['bin_phi1'], df['bin_Genphi1'])
# #print shape of hist_2d
# print(hist_2d.shape)
# print(hist_2d)
# sys.exit()
# # Convert the resulting DataFrame to a numpy array
# hist_2d_array = hist_2d.values


true_samples = df['Genphi1']
observed_samples = df['phi1']
response_hist, _, _ = np.histogram2d(observed_samples, true_samples, bins=bins)
print("\n \n old \n \n")
print(response_hist)
print("\n \n new \n \n")
print(hist_2d_array)

diff = response_hist - hist_2d_array
print(diff)

sys.exit()


data_observed_err = np.sqrt(data_observed)
efficiencies = np.ones_like(data_observed, dtype=float)
efficiencies_err = np.full_like(efficiencies, 0.1, dtype=float)
print(response_hist)
#print size of response hist
print(response_hist.size)
response_hist_err = np.sqrt(response_hist)



fig, ax = plt.subplots()
ax.step(np.arange(num_bins), data_true, where='mid', lw=3,
        alpha=0.7, label='True distribution')
ax.step(np.arange(num_bins), data_observed, where='mid', lw=3,
        alpha=0.7, label='Observed distribution')
ax.set(xlabel='X bins', ylabel='Counts')
ax.legend()


plt.show()




fig, ax = plt.subplots()
im = ax.imshow(response_hist, origin='lower')
cbar = plt.colorbar(im, label='Counts')
ax.set(xlabel='Cause bins', ylabel='Effect bins')
plt.show()


column_sums = response_hist.sum(axis=0)
normalization_factor = efficiencies / column_sums

response = response_hist * normalization_factor
response_err = response_hist_err * normalization_factor

fig, ax = plt.subplots()
im = ax.imshow(response, origin='lower')
cbar = plt.colorbar(im, label='$P(E_i|C_{\mu})$')
ax.set(xlabel='Cause bins', ylabel='Effect bins',
       title='Normalizes response matrix')
plt.show()

unfolded_results = iterative_unfold(data=data_observed,
                                    data_err=data_observed_err,
                                    response=response,
                                    response_err=response_err,
                                    efficiencies=efficiencies,
                                    efficiencies_err=efficiencies_err,
                                    callbacks=[Logger()])

print(unfolded_results.keys())

print(unfolded_results['unfolded'])

print(unfolded_results['sys_err'])


fig, ax = plt.subplots()
ax.step(np.arange(num_bins), data_true, where='mid', lw=3,
        alpha=0.7, label='True distribution')
ax.step(np.arange(num_bins), data_observed, where='mid', lw=3,
        alpha=0.7, label='Observed distribution')
ax.errorbar(np.arange(num_bins), unfolded_results['unfolded'],
            yerr=unfolded_results['sys_err'],
            alpha=0.7,
            elinewidth=3,
            capsize=4,
            ls='None', marker='.', ms=10,
            label='Unfolded distribution')

ax.set(xlabel='X bins', ylabel='Counts')
plt.legend()
plt.show()

sys.exit()

# With PyUnfold, any response matrix should be 2-dimensional where the first dimension (rows) are effect bins and 
# the second dimension (columns) are the causes bins.

"""
In the example outlined above, each sample has a true x and y and an observed x and y. 
Since there are 19 x bins and 9 y bins, altogether there are 171 effects bins and 171 cause bins. 
One possible way to construct the effects bins is to have the first effect bin contain samples 
that are in the first x and first y bin, the second effect bin contains samples that are in the first x bin and second y bin, 
the third effect bin contains samples that are in the first x bin and third y bin, etc. 
A similar scheme could then be used to define the cause bins.
from https://github.com/jrbourbeau/pyunfold/issues/89
'unrolling'
"""

test_file = "/mnt/d/GLOBUS/CLAS12/Thesis/2_selected_dvpip_events/inb/rec/dvpip_events_norad_10000_20230703_1814_Fall_2018_Inbending_50nA_recon.pkl"
binned_test_file = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/rec/singles_t2/binned_dvpip_events_norad_10000_20230703_1814_Fall_2018_Inbending_50nA_recon.pkl"
df = pd.read_pickle(test_file)#.head(1000)
print(df.columns.values)


# for 'xB' and 'GenxB'
# for 'xB' and 'GenxB'
bins_xB_extended = np.concatenate(([-np.inf], bins_xB, [np.inf]))

#print(bins_xB_extended)
#print(bins_xB_extended[:-1])
df['bin_xB'] = pd.cut(df['xB'], bins=bins_xB_extended, right=False, labels=bins_xB_extended[:-1])
df['bin_GenxB'] = pd.cut(df['GenxB'], bins=bins_xB_extended, right=False, labels=bins_xB_extended[:-1])

# for 'Q2' and 'GenQ2'
bins_Q2_extended = np.concatenate(([-np.inf], bins_Q2, [np.inf]))
df['bin_Q2'] = pd.cut(df['Q2'], bins=bins_Q2_extended, right=False, labels=bins_Q2_extended[:-1])
df['bin_GenQ2'] = pd.cut(df['GenQ2'], bins=bins_Q2_extended, right=False, labels=bins_Q2_extended[:-1])

# for 't1' and 'Gent1'
bins_t1_extended = np.concatenate(([-np.inf], bins_t1, [np.inf]))
df['bin_t1'] = pd.cut(df['t1'], bins=bins_t1_extended, right=False, labels=bins_t1_extended[:-1])
df['bin_Gent1'] = pd.cut(df['Gent1'], bins=bins_t1_extended, right=False, labels=bins_t1_extended[:-1])

# for 'phi1' and 'Genphi1'
bins_phi1_extended = np.concatenate(([-np.inf], bins_phi1, [np.inf]))
df['bin_phi1'] = pd.cut(df['phi1'], bins=bins_phi1_extended, right=False, labels=bins_phi1_extended[:-1])
df['bin_Genphi1'] = pd.cut(df['Genphi1'], bins=bins_phi1_extended, right=False, labels=bins_phi1_extended[:-1])


#print xB, bin_xB, GenxB, bin_GenxB
print(df[['xB', 'bin_xB', 'GenxB', 'bin_GenxB']].head(10))
#print Q2, bin_Q2, GenQ2, bin_GenQ2
print(df[['Q2', 'bin_Q2', 'GenQ2', 'bin_GenQ2']].head(10))
#print t1, bin_t1, Gent1, bin_Gent1
print(df[['t1', 'bin_t1', 'Gent1', 'bin_Gent1']].head(10))
#print phi1, bin_phi1, Genphi1, bin_Genphi1
print(df[['phi1', 'bin_phi1', 'Genphi1', 'bin_Genphi1']].head(10))

# create a column with unique bin combination for 'xB', 'Q2', 't1', 'phi1'
df['bin_comb'] = df['bin_xB'].astype(str) + '_' + df['bin_Q2'].astype(str) + '_' + df['bin_t1'].astype(str) + '_' + df['bin_phi1'].astype(str)

# create a column with unique bin combination for 'GenxB', 'GenQ2', 'Gent1', 'Genphi1'
df['bin_comb_Gen'] = df['bin_GenxB'].astype(str) + '_' + df['bin_GenQ2'].astype(str) + '_' + df['bin_Gent1'].astype(str) + '_' + df['bin_Genphi1'].astype(str)

# create a column for unique bin numbers, using pandas' factorize method
df['bin_number'] = pd.factorize(df['bin_comb'])[0]

# similarly for the 'Gen' bin combinations
df['bin_number_Gen'] = pd.factorize(df['bin_comb_Gen'])[0]

print(df['bin_number_Gen'])


# Define all bins arrays
bins_arrays = {'xB': bins_xB_extended, 'Q2': bins_Q2_extended, 't1': bins_t1_extended, 'phi1': bins_phi1_extended}

# Create the bin_number columns for all variables
plot_individual_bins = False
if plot_individual_bins:
    for var in bins_arrays.keys():
        # create a dictionary to map bin edges to bin indices
        bin_dict = {edge: i for i, edge in enumerate(bins_arrays[var][:-1])}
        bin_Gen_dict = {edge: i for i, edge in enumerate(bins_arrays[var][:-1])}

        # create the bin_number columns
        df[f'bin_{var}_number'] = df[f'bin_{var}'].map(bin_dict)
        df[f'bin_Gen{var}_number'] = df[f'bin_Gen{var}'].map(bin_Gen_dict)
        plt.figure(figsize=(10, 8))

        plt.hist2d(df[f'bin_{var}_number'], df[f'bin_Gen{var}_number'], bins=[df[f'bin_{var}_number'].nunique(), df[f'bin_Gen{var}_number'].nunique()])#, norm=mpl.colors.LogNorm())
        plt.colorbar(label='Counts')
        plt.xlabel(f'bin_{var}_number')
        plt.ylabel(f'bin_Gen{var}_number')
        plt.title(f'bin_{var}_number')
        plt.show()
    


# Print the number of unique bin numbers
print("Unique 'bin_number' values:", df['bin_number'].nunique())

# Print the number of unique 'Gen' bin numbers
print("Unique 'bin_number_Gen' values:", df['bin_number_Gen'].nunique())


#create a plot of bin_number vs bin_number_Gen using matplotlib

#plt.scatter(df['bin_number'], df['bin_number_Gen'])
#plt.show()

# Create a 2D histogram of bin_number vs bin_number_Gen
# plot with lognorm virdis
print(df['bin_t1'].unique())
print(df['bin_Gent1'].unique())

plt.figure(figsize=(10, 8))

#print the rows where bin_Gent1 is NaN
print(df[df['bin_Gent1'].isnull()])
#save this row as a csv
df[df['bin_Gent1'].isnull()].to_csv('bin_Gent1_isnull.csv')
#plt.hist2d(df['bin_number'], df['bin_number_Gen'], bins=[df['bin_number'].nunique(), df['bin_number_Gen'].nunique()], norm=mpl.colors.LogNorm())
plt.hist2d(df['bin_number'], df['bin_number_Gen'], bins=[df['bin_number'].nunique(), df['bin_number_Gen'].nunique()], norm=mpl.colors.LogNorm())
plt.colorbar(label='Counts')
plt.xlabel('bin_number')
plt.ylabel('bin_number_Gen')
plt.title('2D Histogram of bin_number vs bin_number_Gen')
plt.show()


