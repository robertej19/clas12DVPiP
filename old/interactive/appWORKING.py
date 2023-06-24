import dash
from dash import dcc
import dash_html_components as html
from dash.dependencies import Input, Output
#import cPickle
#import gZip

import plotly.graph_objects as go
import plotly.io as pio

import plotly.express as px

import matplotlib as mpl
import matplotlib.pyplot as plt

import random 
import sys
import os, subprocess
import argparse
import shutil
import time
from datetime import datetime 
import json

import pandas as pd
import numpy as np
import argparse
import itertools
import os, sys
from icecream import ic
import matplotlib


import filestruct
import numpy as np
from datetime import datetime
import pandas as pd

#import sklearn.externals.joblib as extjoblib
import joblib


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

def outprint(stringx):
    print(stringx)
    now = datetime.now()
    dt_string = now.strftime("%H:%M:%S")
    print(dt_string)

outprint("Loading Data")

server = app.server

Sex = ['M','F']
Sex_Values = ['Male','Female']
Lift_Values = ['Squat','Bench','Deadlift']

equip_avaliable_types = ['Single-ply', 'Multi-ply', 'Wraps', 'Raw','Straps']
equip_types = ['Raw','Equipped','All']

m_weightclasses = [59, 66, 74, 83, 93, 105, 120,128]
f_weighclasses = [47, 52, 57, 63, 72, 84,90]
wclasses =[m_weightclasses,f_weighclasses]


lifts_hist_data = joblib.load("dataArrays/lift_hists.pkl")
hist_bins = joblib.load("dataArrays/hist_bins.pkl")
lifts_labels = joblib.load("dataArrays/hist_lables.pkl")

df_minibin = joblib.load("dataArrays/full_xsection.pkl")
#df_minibin = joblib.load("dataArrays/full_xsection_noseccut.pkl")

fs = filestruct.fs()

q2bins,xBbins, tbins, phibins = [fs.q2bins, fs.xBbins, fs.tbins, fs.phibins]
q2bins = q2bins[:-6]

cmap = plt.cm.jet  # define the colormap


q2bins = [2.5,3.0]
xBbins = [0.3,0.35]
for qmin,qmax in zip(q2bins[0:-1],q2bins[1:]):
    for xmin,xmax in zip(xBbins[0:-1],xBbins[1:]):

        df = df_minibin.query("qmin==@qmin and xmin==@xmin")


        t = df["tmin"].to_numpy()
        p = df["pmin"].to_numpy()

        r = df["xsec_ratio_exp_corr"].to_numpy()
        run = df["uncert_xsec_ratio_exp_corr"].to_numpy()
        n_exp = df["counts_exp"].to_numpy()
        n_exp_uncert = df["uncert_counts_exp"].to_numpy()
        n_rec = df["counts_rec"].to_numpy()
        n_rec_uncert = df["uncert_counts_rec"].to_numpy()
        n_gen = df["counts_gen"].to_numpy()
        n_gen_uncert = df["uncert_counts_gen"].to_numpy()
        clas6_res = df["dsdtdp"].to_numpy()
        df["uncert_dsdtdp"] = np.sqrt( np.square(df["stat"]/df["dsdtdp"]) + np.square(df["sys"]/df["dsdtdp"]))*df["dsdtdp"]
        clas6_res_uncert = df["uncert_dsdtdp"].to_numpy()
        n_12GeV = df["counts_10600GeV"].to_numpy()
        n_6GeV = df["counts_5776GeV"].to_numpy()
        xsec_ratio_sim = df["xsec_ratio_sim"].to_numpy()
        df["uncert_xsec_ratio_sim"] = np.sqrt( np.square(df["uncert_counts_10600GeV"]/df["counts_10600GeV"]) + np.square(df["uncert_counts_5776GeV"]/df["counts_5776GeV"])) * df["xsec_ratio_sim"]
        xsec_ratio_sim_uncert = df["uncert_xsec_ratio_sim"].to_numpy()
        acc_corr = (1/df["acc_corr"]).to_numpy()
        acc_corr_uncert = (df["uncert_acc_corr"]/df["acc_corr"]*(1/df["acc_corr"])).to_numpy()


        x = np.reshape(p, (len(tbins)-1, len(phibins)-1))
        y = np.reshape(t, (len(tbins)-1, len(phibins)-1))
        z = np.reshape(r, (len(tbins)-1, len(phibins)-1))
        zuncert = np.reshape(run, (len(tbins)-1, len(phibins)-1))
        n_exp_shaped = np.reshape(n_exp, (len(tbins)-1, len(phibins)-1))
        n_exp_uncert_shaped = np.reshape(n_exp_uncert, (len(tbins)-1, len(phibins)-1))
        n_rec_shaped = np.reshape(n_rec, (len(tbins)-1, len(phibins)-1))
        n_rec_uncert_shaped = np.reshape(n_rec_uncert, (len(tbins)-1, len(phibins)-1))
        n_gen_shaped = np.reshape(n_gen, (len(tbins)-1, len(phibins)-1))
        n_gen_uncert_shaped = np.reshape(n_gen_uncert, (len(tbins)-1, len(phibins)-1))
        clas6_res_shaped = np.reshape(clas6_res, (len(tbins)-1, len(phibins)-1))
        clas6_res_uncert_shaped = np.reshape(clas6_res_uncert, (len(tbins)-1, len(phibins)-1))
        n_12GeV_shaped = np.reshape(n_12GeV, (len(tbins)-1, len(phibins)-1))
        n_6GeV_shaped = np.reshape(n_6GeV, (len(tbins)-1, len(phibins)-1))
        xsec_ratio_sim_shaped = np.reshape(xsec_ratio_sim, (len(tbins)-1, len(phibins)-1))
        xsec_ratio_sim_uncert_shaped = np.reshape(xsec_ratio_sim_uncert, (len(tbins)-1, len(phibins)-1))
        acc_corr_shaped = np.reshape(acc_corr, (len(tbins)-1, len(phibins)-1))
        acc_corr_uncert_shaped = np.reshape(acc_corr_uncert, (len(tbins)-1, len(phibins)-1))



        z = np.ma.masked_where(z==0, z)

        cmap.set_bad(color='white')

""" Columns: 
Index(['qmin', 'xmin', 'tmin', 'pmin', 'qave_exp', 'xave_exp', 'tave_exp',
       'pave_exp', 'counts_exp', 'qave_rec', 'xave_rec', 'tave_rec',
       'pave_rec', 'counts_rec', 'qave_gen', 'xave_gen', 'tave_gen',
       'pave_gen', 'counts_gen', 'counts_10600GeV', 'counts_5776GeV',
       'Energy_ratio', 'q', 'x', 't', 'p', 'dsdtdp', 'stat', 'sys', 'qmax',
       'xmax', 'tmax', 'pmax', 'gamma_exp', 'epsi_exp', 'gamma6_sim',
       'gamma12_sim', 'xsec_sim_12', 'xsec_sim_6', 'xsec_ratio_sim', 'binvol',
       'acc_corr', 'xsec', 'xsec_corr', 'xsec_corr_red', 'xsec_corr_red_nb',
       'xsec_ratio_exp', 'xsec_ratio_exp_corr', 'uncert_counts_exp',
       'uncert_counts_rec', 'uncert_counts_gen', 'uncert_counts_10600GeV',
       'uncert_counts_5776GeV', 'uncert_xsec', 'uncert_acc_corr',
       'uncert_xsec_corr_red_nb', 'uncert_xsec_ratio_exp',
       'uncert_xsec_ratio_exp_corr'],
      dtype='object')
"""


xx = x[0]
yy = y[:,0]
np.append(yy,10)
xxx = xx + 9
yyy = []

for ymin,ymax in zip(yy[:-1],yy[1:]):
    yyy.append((ymin+ymax)/2)

globalblue = '#1670b5'

print(z)
figXXX = go.Figure(data=go.Heatmap(
                   z=z,
                   x=xxx,
                   y=yyy,
                   hovertemplate='Lepton-Hadron Angle: %{x}<br>-t (GeV^2) %{y}<br>Cross Section Ratio: %{z} $\pm$ %{zuncert} <extra></extra>',
                   hoverongaps = False))

axis_y = dict(range = [0,2])#, autorange = False,
             #showgrid = False, zeroline = False,
             #linecolor = 'black')#, showticklabels = False,
             #ticks = '' )

axis_x = dict(range = [0,360])#, autorange = False,
            # showgrid = False, zeroline = False,
            # linecolor = 'black')#, showticklabels = False,
             #ticks = '' )

figXXX.update_layout(margin = dict(t=200,r=200,b=200,l=200),
    xaxis = axis_x,
    yaxis = axis_y,
    showlegend = False,
    width = 1800, height = 1200,
    autosize = False )



newdats = np.stack((zuncert,
        n_exp_shaped,n_exp_uncert_shaped,
        n_rec_shaped,n_rec_uncert_shaped ,
        n_gen_shaped,n_gen_uncert_shaped ,
        clas6_res_shaped,clas6_res_uncert_shaped ,
        n_12GeV_shaped , n_6GeV_shaped,  
        xsec_ratio_sim_shaped, xsec_ratio_sim_uncert_shaped,
        acc_corr_shaped,acc_corr_uncert_shaped),axis=-1)
 

figXXX.update_traces(customdata=newdats,
         hovertemplate = '<b>CLAS12:CLAS6 Corrected Reduced Cross Section: %{z:.2f} +/- %{customdata[0]:.2f}<br>'
                '<b>Lepton-Hadron Angle= %{x:.0f}, -t= %{y:.2f}<br>'
                '<b>Raw F18 In Counts:  %{customdata[1]:.2f} +/- %{customdata[2]:.2f}<br>'+
                '<b>N_Gen_Norad:  %{customdata[5]:.2f} +/- %{customdata[6]:.2f}<br>'+
                '<b>N_Rec_Norad:  %{customdata[3]:.2f} +/- %{customdata[4]:.2f}<br>'+
                '<b>1/Acc_Corr:  %{customdata[13]:.2f} +/- %{customdata[14]:.2f}<br>'+
                '<b>CLAS6 Result:  %{customdata[7]:.2f} +/- %{customdata[8]:.2f}<br>'+
                '<b>N_10.6GeV_sim  %{customdata[9]:.2f} <br>'+
                '<b>N_5.75GeV_sim %{customdata[10]:.2f}<br>'+
                '<b>10.6GeV:5.75GeV Cross Section Ratio:  %{customdata[11]:.2f} +/- %{customdata[12]:.2f}<br> <extra></extra>',
                
                #'test string {}'.format(x),
                text = ["this is text"],
                hoverlabel = {
                'bgcolor': globalblue,
                'font': {'color':'white'}
                }
                )
 


Sex_Values = ["Reduced Corrected Cross Section Ratio","Reduced Cross Section Ratio","CLAS12 Data","CLAS12 Cross Section"]
outprint("Plotting Data")

"""
    html.Div([
                dcc.RadioItems(
                    id='sex-type',
                    options=[{'label': i, 'value': i} for i in Sex_Values],
                    value=Sex_Values[0],
                    labelStyle={'display': 'inline-block'}
                )
            ],
            style={'width': '20%', 'padding-left':'10%', 'display': 'inline-block'}),
 
"""

app.layout = html.Div([
    html.Div([



    dcc.Graph(id='indicator-graphic',figure=figXXX, config={'displayModeBar':False}),

    html.Label('Min. x_B', id='male-weight-class'),

    html.Div(
        html.Div(
            dcc.Slider(
                id='male-year--slider',
                min=0.1,
                max=0.6,
                value=0.3,
                marks={
                    0.1:"0.1",
                    0.15:"0.15",
                    0.2:"0.2",
                    0.25:"0.25",
                    0.3:"0.3",
                    0.35:"0.35",
                    0.4: "0.4",
                    0.45:"0.45",
                    0.5:"0.5",
                    0.55:"0.55",
                    0.6:"0.6",
                },
                step=None,
                updatemode='drag',
            ),
            id='malesliderContainer'),
        style={'width': '60%','padding-left':'10%', 'padding-right':'10%'}

        ),

#self.xBbins = [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.7,0.85,1]#
#		self.q2bins =  [1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,7.0,9.0,14.0]

    html.Label('Min. Q^2 (GeV^2)', id='female-weight-class'),
    html.Div(
        html.Div(
            dcc.Slider(
                id='female-year--slider',
                min=1,
                max=4.5,
                value=1.5,
                marks={
                        1 : '1 GeV^2',
                        1.5 : '1.5',
                        2.0 : '2',
                        2.5 : '2.5',
                        3.0 : '3.0',
                        3.5 : '3.5',
                        4.0 : '4.0',
                        4.5 : '4.5',
                },
                step=None,
                updatemode='drag',
            ),
            id='femalesliderContainer'),
        style={'width': '60%','padding-left':'10%', 'padding-right':'10%'}

        )

    
])
])



@app.callback(
    Output('indicator-graphic', 'figure'),
    [ Input('indicator-graphic', 'hoverData'),
    Input('female-year--slider', 'value'),
    Input('male-year--slider', 'value')])

def update_graph_old(hoverdata,q2_value,xb_value):

            print("HERE IS THE SEX VAL {} {} {}".format(hoverdata,q2_value,xb_value))

            qmin = q2_value
            xmin = xb_value#xb_value
            print(df_minibin)
            queri = "qmin=={} and xmin=={}".format(qmin,xmin)
            print(queri)
            df = df_minibin.query(queri)
            print(df)

            t = df["tmin"].to_numpy()
            p = df["pmin"].to_numpy()

            r = df["xsec_ratio_exp_corr"].to_numpy()
            run = df["uncert_xsec_ratio_exp_corr"].to_numpy()
            n_exp = df["counts_exp"].to_numpy()
            n_exp_uncert = df["uncert_counts_exp"].to_numpy()
            n_rec = df["counts_rec"].to_numpy()
            n_rec_uncert = df["uncert_counts_rec"].to_numpy()
            n_gen = df["counts_gen"].to_numpy()
            n_gen_uncert = df["uncert_counts_gen"].to_numpy()
            clas6_res = df["dsdtdp"].to_numpy()
            df["uncert_dsdtdp"] = np.sqrt( np.square(df["stat"]/df["dsdtdp"]) + np.square(df["sys"]/df["dsdtdp"]))*df["dsdtdp"]
            clas6_res_uncert = df["uncert_dsdtdp"].to_numpy()
            n_12GeV = df["counts_10600GeV"].to_numpy()
            n_6GeV = df["counts_5776GeV"].to_numpy()
            xsec_ratio_sim = df["xsec_ratio_sim"].to_numpy()
            df["uncert_xsec_ratio_sim"] = np.sqrt( np.square(df["uncert_counts_10600GeV"]/df["counts_10600GeV"]) + np.square(df["uncert_counts_5776GeV"]/df["counts_5776GeV"])) * df["xsec_ratio_sim"]
            xsec_ratio_sim_uncert = df["uncert_xsec_ratio_sim"].to_numpy()
            acc_corr = (1/df["acc_corr"]).to_numpy()
            acc_corr_uncert = (df["uncert_acc_corr"]/df["acc_corr"]*(1/df["acc_corr"])).to_numpy()
            xsec_corr_red_nb = df["xsec_corr_red_nb"].to_numpy()
            uncert_xsec_corr_red_nb = df["uncert_xsec_corr_red_nb"].to_numpy()

            x = np.reshape(p, (len(tbins)-1, len(phibins)-1))
            y = np.reshape(t, (len(tbins)-1, len(phibins)-1))
            z = np.reshape(r, (len(tbins)-1, len(phibins)-1))
            zuncert = np.reshape(run, (len(tbins)-1, len(phibins)-1))
            n_exp_shaped = np.reshape(n_exp, (len(tbins)-1, len(phibins)-1))
            n_exp_uncert_shaped = np.reshape(n_exp_uncert, (len(tbins)-1, len(phibins)-1))
            n_rec_shaped = np.reshape(n_rec, (len(tbins)-1, len(phibins)-1))
            n_rec_uncert_shaped = np.reshape(n_rec_uncert, (len(tbins)-1, len(phibins)-1))
            n_gen_shaped = np.reshape(n_gen, (len(tbins)-1, len(phibins)-1))
            n_gen_uncert_shaped = np.reshape(n_gen_uncert, (len(tbins)-1, len(phibins)-1))
            clas6_res_shaped = np.reshape(clas6_res, (len(tbins)-1, len(phibins)-1))
            clas6_res_uncert_shaped = np.reshape(clas6_res_uncert, (len(tbins)-1, len(phibins)-1))
            n_12GeV_shaped = np.reshape(n_12GeV, (len(tbins)-1, len(phibins)-1))
            n_6GeV_shaped = np.reshape(n_6GeV, (len(tbins)-1, len(phibins)-1))
            xsec_ratio_sim_shaped = np.reshape(xsec_ratio_sim, (len(tbins)-1, len(phibins)-1))
            xsec_ratio_sim_uncert_shaped = np.reshape(xsec_ratio_sim_uncert, (len(tbins)-1, len(phibins)-1))
            acc_corr_shaped = np.reshape(acc_corr, (len(tbins)-1, len(phibins)-1))
            acc_corr_uncert_shaped = np.reshape(acc_corr_uncert, (len(tbins)-1, len(phibins)-1))
            xsec_corr_red_nb_shaped = np.reshape(xsec_corr_red_nb, (len(tbins)-1, len(phibins)-1))
            uncert_xsec_corr_red_nb_shaped = np.reshape(uncert_xsec_corr_red_nb, (len(tbins)-1, len(phibins)-1))


            z = np.ma.masked_where(z==0, z)

            cmap.set_bad(color='white')



            newdats = np.stack((zuncert,
                    n_exp_shaped,n_exp_uncert_shaped,
                    n_rec_shaped,n_rec_uncert_shaped ,
                    n_gen_shaped,n_gen_uncert_shaped ,
                    clas6_res_shaped,clas6_res_uncert_shaped ,
                    n_12GeV_shaped , n_6GeV_shaped,  
                    xsec_ratio_sim_shaped, xsec_ratio_sim_uncert_shaped,
                    acc_corr_shaped,acc_corr_uncert_shaped,
                    xsec_corr_red_nb_shaped,uncert_xsec_corr_red_nb_shaped),axis=-1)    
        

        
        
                     
            fig10 = {
                    'data' : [go.Heatmap(
                            z=z,
                            x=xxx,
                            y=yyy,
                            zmin=0.1, 
                            zmax=1.4,
                            customdata=newdats,
                            hovertemplate = '<b>Ratio of CLAS12 to CLAS6 Energy Corrected Reduced Cross Sections: %{z:.2f} +/- %{customdata[0]:.2f}<br>'+
                                    '<b> <br>'+
                                    '<b>Raw F18_In Counts:  %{customdata[1]:.0f} +/- %{customdata[2]:.0f}<br>'+
                                    '<b>N_sim_gen:  %{customdata[5]:.0f} +/- %{customdata[6]:.0f}<br>'+
                                    '<b>N_sim_rec:  %{customdata[3]:.0f} +/- %{customdata[4]:.0f}<br>'+
                                    '<b>Acc_Corr:  %{customdata[13]:.0f} +/- %{customdata[14]:.0f}<br>'+
                                    '<b> <br>'+
                                    '<b>CLAS12 Reduced Cross Section:  %{customdata[15]:.2f} nb +/- %{customdata[16]:.1f} nb<br>'+
                                    '<b>CLAS6 Reduced Cross Section:  %{customdata[7]:.2f} nb +/- %{customdata[8]:.2f} nb <br>'+
                                    '<b> <br>'+
                                    '<b>N_gen_10.6_GeV:  %{customdata[9]:.0f}<br>'
                                    '<b>N_gen_5.75_GeV: %{customdata[10]:.0f}<br>'+
                                    '<b>Beam Energy Correction Factor:  %{customdata[11]:.2f} +/- %{customdata[12]:.2f}<br> <extra></extra>',
                                    
                                    #'test string {}'.format(x),
                                    text = ["this is text"],
                                    hoverlabel = {
                                    'bgcolor': globalblue,
                                    'font': {'color':'white'}
                                    }
                            )


                            # hoverongaps = False))
                            ],
                    'layout': dict(
                        title='<b> Ratio of CLAS12 to CLAS6 Reduced Cross Sections, Beam Energy Corrected, Q^2 =' +str(q2_value)+', x_B = '+ str(xb_value) +'<br>',
                        xaxis={
                            'range': [0,360],
                            'title': 'lepton-hadron angle',
                            'textfont' : dict(
                                family="Georgia",
                                size=360,
                                #color="#7f7f7f"
                             ),
                            'tickvals' : [45,90,135,180,225,270,315],
                            'ticktext' : ['45','90','135','180','225','270','315']
                        },
                        yaxis={
                            'range': [0,2],
                            'title': '-t (GeV^2)',
                            'textfont' : dict(
                                family="Georgia",
                                size=360,
                                #color="#7f7f7f"
                            ),
                        },

                        #hovertext=["Text A", "Text B", "Text C", "Text D", "Text E"],
                        #hoverinfo="text",
                        margin={'l': 100, 'b': 200, 't': 200, 'r': 100},
                        #hovermode='x unified',
                    )
                }

            return fig10




"""
@app.callback(Output('indicator-graphic', 'layout'),
    [Input('indicator-graphic', 'hoverData')])
def update_slide(hoverdata):



    x_val = hoverdata['points']
    #print(type(x_val))
    xx = x_val[0]
    x_ind = xx['pointNumber']

    colors = ['#1670b5',]*100
    #colors[25] = '#59e94a' #green
    colors[25] = '#e9524a' #red

    marker_color = colors

    return
"""
    #print(xx_val)
    #if ind == 0:
    #    return {}
    #else:
    #    print(hoverdata)
        #return {'display': 'none'}


@app.callback(Output('female-weight-class', 'style'),
    [Input('sex-type', 'value')])
def update_slide(sex_val):

    ind = Sex_Values.index(sex_val)

    if ind == 1:
        return {'width': '15%','padding-left':'10%', 'display': 'inline-block'}
    else:
        return {'display': 'none'}

@app.callback(Output('male-weight-class', 'style'),
    [Input('sex-type', 'value')])
def update_slide(sex_val):

    ind = Sex_Values.index(sex_val)

    if ind == 0:
        return {'width': '15%','padding-left':'10%', 'display': 'inline-block'}
    else:
        return {'display': 'none'}

if __name__ == '__main__':
    app.run_server(debug=False)
