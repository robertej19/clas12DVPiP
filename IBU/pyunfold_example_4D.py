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
total_bins = (len(bins_Q2)-1)*(len(bins_xB)-1)*(len(bins_t1)-1)*(len(bins_phi1)-1)
#create an unrolled array:
bins_unrolled = np.arange(total_bins)
print(len(bins_unrolled))

def unroller_Gen(row, bins_Q2, bins_xB, bins_t1, bins_phi1):
    """Unroll 4D bin numbers into 1D bin number"""
    bin_xB, bin_Q2, bin_t1, bin_phi1 = row['bin_GenxB'], row['bin_GenQ2'], row['bin_Gent1'], row['bin_Genphi1']

    # get index of bins_xB where bin_xB is
    xb_index = np.digitize(bin_xB, bins_xB, right=False) - 1
    # get index of bins_Q2 where bin_Q2 is
    Q2_index = np.digitize(bin_Q2, bins_Q2, right=False) - 1
    # get index of bins_t1 where bin_t1 is
    t1_index = np.digitize(bin_t1, bins_t1, right=False) - 1
    # get index of bins_phi1 where bin_phi1 is
    phi1_index = np.digitize(bin_phi1, bins_phi1, right=False) - 1

    # return the unrolled bin number
    return xb_index * (len(bins_Q2) - 1) * (len(bins_t1) - 1) * (len(bins_phi1) - 1) + \
           Q2_index * (len(bins_t1) - 1) * (len(bins_phi1) - 1) + \
           t1_index * (len(bins_phi1) - 1) + \
           phi1_index

#df['truth_bin_id'] = df.apply(unroller_Gen, axis=1, args=(bins_Q2, bins_xB, bins_t1, bins_phi1))


# #iterate over all but last value in bins_xB
# for x in bins_xB[:-1]:
#         for q in bins_Q2[:-1]:
#                 for t in bins_t1[:-1]:
#                         for p in bins_phi1[:-1]:
#                                 print(x,q,t,p)
#                                 print(unroller_Gen(x,q,t,p, bins_Q2, bins_xB, bins_t1, bins_phi1))
# sys.exit()g

# Need to wrap on phi!
def unroller(row, bins_Q2, bins_xB, bins_t1, bins_phi1):
        """Unroll 4D bin numbers into 1D bin number"""

        bin_xB, bin_Q2, bin_t1, bin_phi1 = row['bin_xB'], row['bin_Q2'], row['bin_t1'], row['bin_phi1']
        #bin_xB, bin_Q2, bin_t1, bin_phi1 = x,q,t,p
        
        #get index of bins_xB where bin_xB is
        # get index of bins_xB where bin_xB is
        xb_index = np.digitize(bin_xB, bins_xB, right=False) - 1
        # get index of bins_Q2 where bin_Q2 is
        Q2_index = np.digitize(bin_Q2, bins_Q2, right=False) - 1
        # get index of bins_t1 where bin_t1 is
        t1_index = np.digitize(bin_t1, bins_t1, right=False) - 1
        # get index of bins_phi1 where bin_phi1 is
        phi1_index = np.digitize(bin_phi1, bins_phi1, right=False) - 1

        #return the unrolled bin number
        return xb_index * (len(bins_Q2)-1) * (len(bins_t1)-1) * (len(bins_phi1)-1) + \
                Q2_index * (len(bins_t1)-1) * (len(bins_phi1)-1) + \
                t1_index * (len(bins_phi1)-1) + \
                phi1_index


uniform_prior = uniform_prior(num_causes=4)
causes = np.arange(4)
jeffreys_prior = jeffreys_prior(causes=causes)

test_file = "/mnt/d/GLOBUS/CLAS12/Thesis/2_selected_dvpip_events/inb/rec/dvpip_events_norad_10000_20230703_1814_Fall_2018_Inbending_50nA_recon.pkl"
binned_test_file = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/rec/singles_t2/binned_dvpip_events_norad_10000_20230703_1814_Fall_2018_Inbending_50nA_recon.pkl"
binned_test_file_prepped = "binned_test_file_prepped_4x4x4xphi.pkl"

data_dir = "/mnt/d/GLOBUS/CLAS12/Thesis/2_selected_dvpip_events/inb/rec/"
read_in = False
if read_in:
        ## for each pickle file in datadir, read it in, and then combine into one dataframe
        # df = pd.DataFrame()
        # for file in os.listdir(data_dir):
        #         if file.endswith(".pkl"):
        #                 print(file)
        #                 df = df.append(pd.read_pickle(data_dir+file), ignore_index=True)
        df = pd.read_pickle(test_file)#.head(300_00)
        #remove all rows where Gent1 is larger than 1.8

        cutoff = 1.7
        cutoff_low = 0.1
        df = df[df['Gent1'] < cutoff]
        df = df[df['t1']<cutoff]
        df = df[df['t1']>cutoff_low]
        df = df[df['Gent1']>cutoff_low]
        #requre phi to be between 0 and 360
        df = df[df['phi1']<360]
        df = df[df['phi1']>0]
        #require Genphi to be between 0 and 360
        df = df[df['Genphi1']<360]
        df = df[df['Genphi1']>0]
        #require xB to be between 0 and 1
        df = df[df['xB']<.8]
        df = df[df['xB']>0]
        #require GenxB to be between 0 and 1
        df = df[df['GenxB']<.8]
        df = df[df['GenxB']>0]
        #require Q2 to be between 0 and 10
        df = df[df['Q2']<11]
        df = df[df['Q2']>1]
        #require GenQ2 to be between 0 and 10
        df = df[df['GenQ2']<11]
        df = df[df['GenQ2']>1]


        sns.set_context(context='poster')
        plt.rcParams['figure.figsize'] = (10, 8)
        plt.rcParams['lines.markeredgewidth'] = 2

        # for 'xB' and 'GenxB'
        # for 'xB' and 'GenxB'
        bins_xB_extended = np.concatenate(([-10], bins_xB, [30]))

        #print(bins_xB_extended)
        #print(bins_xB_extended[:-1])
        df['bin_xB'] = pd.cut(df['xB'], bins=bins_xB, right=False, labels=bins_xB[:-1])
        df['bin_GenxB'] = pd.cut(df['GenxB'], bins=bins_xB, right=False, labels=bins_xB[:-1])

        # for 'Q2' and 'GenQ2'
        bins_Q2_extended = np.concatenate(([-30], bins_Q2, [30]))
        df['bin_Q2'] = pd.cut(df['Q2'], bins=bins_Q2, right=False, labels=bins_Q2[:-1])
        df['bin_GenQ2'] = pd.cut(df['GenQ2'], bins=bins_Q2, right=False, labels=bins_Q2[:-1])

        # for 't1' and 'Gent1'
        df['bin_t1'] = pd.cut(df['t1'], bins=bins_t1, right=False, labels=bins_t1[:-1])
        df['bin_Gent1'] = pd.cut(df['Gent1'], bins=bins_t1, right=False, labels=bins_t1[:-1])

        # for 'phi1' and 'Genphi1'
        df['bin_phi1'] = pd.cut(df['phi1'], bins=bins_phi1, right=False, labels=bins_phi1[:-1])
        df['bin_Genphi1'] = pd.cut(df['Genphi1'], bins=bins_phi1, right=False, labels=bins_phi1[:-1])


        df = df[(df['bin_t1'].isin(bins_t1[:-1])) & (df['bin_Gent1'].isin(bins_t1[:-1]))]
        df = df[(df['bin_phi1'].isin(bins_phi1[:-1])) & (df['bin_Genphi1'].isin(bins_phi1[:-1]))]
        #do also for xB and Q2
        df = df[(df['bin_xB'].isin(bins_xB[:-1])) & (df['bin_GenxB'].isin(bins_xB[:-1]))]
        df = df[(df['bin_Q2'].isin(bins_Q2[:-1])) & (df['bin_GenQ2'].isin(bins_Q2[:-1]))]



        # Now use apply() to apply unroller() to every row of df
        df['observed_bin_id'] = df.apply(unroller, axis=1, args=(bins_Q2, bins_xB, bins_t1, bins_phi1))
        print("finished unrolling observed bins")
        df['truth_bin_id'] = df.apply(unroller_Gen, axis=1, args=(bins_Q2, bins_xB, bins_t1, bins_phi1))
        print("finished unrolling truth bins")


        print(df['truth_bin_id'])
        print(df['observed_bin_id'])     


        print(df['truth_bin_id'].nunique())
        print(df['observed_bin_id'].nunique())

        df.to_pickle(binned_test_file_prepped)
else:
        df = pd.read_pickle(binned_test_file_prepped)

#get the unique bins in bin_number
unique_bins = bins_unrolled
print(unique_bins)
phi = False
t1 = False
if phi:
        bins = bins_phi1
        binned_truth = 'bin_Genphi1'
        binned_data = 'bin_phi1'
        num_bins = len(bins)-1
elif t1: 
        bins = bins_t1
        binned_truth = 'bin_Gent1'
        binned_data = 'bin_t1'
        num_bins = len(bins)-1
else:
        pass
print("using unrolled bins")
#append a bin to the end of bins as a dummy WARNING: MIGHT CAUSE ISSUES
binned_truth = 'truth_bin_id_remapped'
binned_data = 'observed_bin_id_remapped'
#bins = unique_bins
unique_truth_bin_id = sorted(df['truth_bin_id'].unique())
# Create a dictionary that maps unique values in 'truth_bin_id' to a sequence of integers
mapping = {value: i for i, value in enumerate(unique_truth_bin_id)}

# Create the new column 'truth_bin_id_remapped' using the mapping
df['truth_bin_id_remapped'] = df['truth_bin_id'].map(mapping)
df['observed_bin_id_remapped'] = df['observed_bin_id'].map(mapping)

new_bins = sorted(df['truth_bin_id_remapped'].unique())
bins = new_bins
bins = np.append(bins, bins[-1]+1)
num_bins = len(bins)-1
        

value_counts_truth = df[binned_truth].value_counts()

counts_series = pd.Series(0, index=bins[:-1])
counts_truth = counts_series.add(value_counts_truth, fill_value=0)
data_true = counts_truth.values

value_counts_observed= df[binned_data].value_counts()
counts_series = pd.Series(0, index=bins[:-1])
counts_observed = counts_series.add(value_counts_observed, fill_value=0)
data_observed = counts_observed.values

# Creating a zero-filled DataFrame with desired index and columns
hist_2d = pd.DataFrame(0, index=bins[:-1], columns=bins[:-1])

# Calculating the counts using crosstab
counts = pd.crosstab(df[binned_data], df[binned_truth])

# Filling the actual counts into our zero-filled DataFrame
hist_2d.update(counts)

# Converting the DataFrame to numpy array
response_hist = hist_2d.values
#print the size of the array
print("response_hist shape")
print(response_hist.shape)


data_observed_err = np.sqrt(data_observed)
efficiencies = np.ones_like(data_observed, dtype=float)
efficiencies_err = np.full_like(efficiencies, 0.1, dtype=float)
response_hist_err = np.sqrt(response_hist)

column_sums = response_hist.sum(axis=0)
# Add a small constant to avoid division by zero
column_sums = column_sums + 1e-10
normalization_factor = efficiencies / column_sums

response = response_hist * normalization_factor
response_err = response_hist_err * normalization_factor


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



fig, ax = plt.subplots()
im = ax.imshow(response, origin='lower')
cbar = plt.colorbar(im, label='$P(E_i|C_{\mu})$')
ax.set(xlabel='Cause bins', ylabel='Effect bins',
       title='Normalized response matrix')
plt.show()

unfolded_results = iterative_unfold(data=data_observed,
                                    data_err=data_observed_err,
                                    response=response,
                                    response_err=response_err,
                                    efficiencies=efficiencies,
                                    efficiencies_err=efficiencies_err,
                                    callbacks=[Logger()],
                                    ts_stopping=0.00005,)

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

#create another plot for residuals
fig, ax = plt.subplots()
ax.errorbar(np.arange(num_bins), np.abs(data_observed-data_true),
                alpha=0.7,
                elinewidth=3,
                capsize=4,
                ls='None', marker='.', ms=10,
                label='Observed distribution')
ax.errorbar(np.arange(num_bins), np.abs(unfolded_results['unfolded']-data_true),
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


