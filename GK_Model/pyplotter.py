import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('cross_section_pi0_575.txt', sep='\t', header=0)
#df = pd.read_csv('cross_section_pi0_10600.txt', sep='\t', header=0)


print(df.columns)
base_query = "Q2==2.25 and xB==0.225"
df_small = df.query(base_query)

df_small['sigma_T'] = pd.to_numeric(df["sigma_T"], errors='coerce')
df_small['sigma_L'] = pd.to_numeric(df["sigma_L"], errors='coerce')
df_small['sigma_LT'] = pd.to_numeric(df["sigma_LT"], errors='coerce')
df_small['sigma_TT'] = pd.to_numeric(df["sigma_TT"], errors='coerce')


print(df_small)
df_small.dropna()
print(df_small)


plt.plot(-1*df_small['mt'], df_small['sigma_T'],'o')
plt.plot(-1*df_small['mt'], df_small['sigma_L'],'+')
plt.plot(-1*df_small['mt'], df_small['sigma_LT'],'ro')
plt.plot(-1*df_small['mt'], df_small['sigma_TT'],'r+')




df = pd.read_csv('cross_section_pi0_10600.txt', sep='\t', header=0)


print(df.columns)

df_small = df.query(base_query)

df_small['sigma_T'] = pd.to_numeric(df["sigma_T"], errors='coerce')
df_small['sigma_L'] = pd.to_numeric(df["sigma_L"], errors='coerce')
df_small['sigma_LT'] = pd.to_numeric(df["sigma_LT"], errors='coerce')
df_small['sigma_TT'] = pd.to_numeric(df["sigma_TT"], errors='coerce')


print(df_small)
df_small.dropna()
print(df_small)

#fig, ax = plt.subplots(figsize=(12, 6))

plt.plot(-1*df_small['mt'], df_small['sigma_T'],'k')
plt.plot(-1*df_small['mt'], df_small['sigma_L'],'k')
plt.plot(-1*df_small['mt'], df_small['sigma_LT'],'b')
plt.plot(-1*df_small['mt'], df_small['sigma_TT'],'b')





plt.ylim([-300,400])
plt.xlim([0,2])


plt.show()

print(df_small)