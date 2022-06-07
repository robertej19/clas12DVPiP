import pandas as pd

def get_integrated_lumi(df,bad_runs_list=[]):

    #df = pd.read_pickle("pickled_data/f18_out_with_logistics.pkl")
    #df = pd.read_pickle("pickled_data/f18_inbending_with_logi.pkl")
    #df = pd.read_pickle("pickled_data/exp_outbending_183_with_logi.pkl")

    skipped_runs = []
    total_beam_q = 0
    for run_num in df.RunNum.unique():
        print("On run number {}".format(run_num))
        if run_num in bad_runs_list:
            skipped_runs.append(run_num)
        else:
            run_q = df.query('RunNum == {}'.format(run_num)).beamQ.max()-df.query('RunNum == {}'.format(run_num)).beamQ.min()
            total_beam_q+=run_q

    #print(total_beam_q) 
    # Observe  1065038.810546875 for outbending small sample size
    # observe  42909169.32714844 for inbendining
    # observe  35441123.45703125 for outbending large sample size

    N_a = 6E23
    e = 1.6E-19
    l = 5
    rho = 0.07
    units_conversion_factor = 1E-9

    Lumi = N_a*l*rho*total_beam_q/e*units_conversion_factor

    return Lumi,total_beam_q,skipped_runs #integrated luminosity in units of inverse cm^2

if __name__ == "__main__":
    df = pd.read_pickle("pickled_data/exp_outbending_183_with_logi.pkl")
    lumi,total_beam_q,skipped_runs = get_integrated_lumi(df,bad_runs_list=[5434,5641])
    print(lumi)
    print(skipped_runs)