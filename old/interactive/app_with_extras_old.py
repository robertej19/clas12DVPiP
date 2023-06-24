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
        acc_corr = df["acc_corr"].to_numpy()
        acc_corr_uncert = df["uncert_acc_corr"].to_numpy()


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


outprint("Finished Loading Data")

"""
International Powerlifting Federation (IPF) weight classes:
Women: 47 kg, 52 kg, 57 kg, 63 kg, 72 kg, 84 kg, 84 kg+
Men: 59 kg, 66 kg, 74 kg, 83 kg, 93 kg, 105 kg, 120 kg, 120 kg+
Sex
equipment
age
age class
bodyweight
weightclass
Best3SquatKg
Best3BenchKg
Best3DeadliftKg
TotalKg

--- analytics of powerlifting ---
Dots
Wilks
Glossbrenner
Goodlift

make a histogram - male, weightclass, distro of deadlifts

"""


# # sex_val = 'Male'
# # lift_val = 'Squat'
# # equip_val = 'Raw'
# # male_year_value = 66

# # if Sex_Values.index(sex_val)==0:
# #     year_value = male_year_value
# # else:
# #     year_value = fe_year_value

# # equip_ind = equip_types.index(equip_val)
# # sex_ind = Sex_Values.index(sex_val)
# # weight_ind = wclasses[Sex_Values.index(sex_val)].index(year_value)
# # lift_ind = Lift_Values.index(lift_val)
# # #hist_data = lift_set[Sex_Values.index(sex_val)][wclasses[Sex_Values.index(sex_val)].index(year_value)]


# # hist_x_data = hist_bins
# # hist_y_data = lifts_hist_data[lift_ind][sex_ind][weight_ind][equip_ind]
# # hist_hover_labels = lifts_labels[lift_ind][sex_ind][weight_ind][equip_ind]

# # hist_lb_data = [i*2.20462 for i in hist_x_data]
# # hist_lb_low_data =  [(i-11) for i in hist_lb_data]
# # #print(hist_hover_labels)

# # arr1 = np.array(hist_lb_data)
# # arr2 = np.array(hist_lb_low_data)


# # # # fig, ax = plt.subplots(figsize =(36, 17)) 

# # # # #plt.rcParams["font.family"] = "Times New Roman"
# # # # plt.pcolormesh(x,y,z,cmap=cmap)#colorsx,cmap=cmap)#norm=mpl.colors.LogNorm())

# # # # #plt.pcolormesh(x,y,colors_reshaped,cmap=cmap)#,norm=mpl.colors.LogNorm())

# # # # plt.clim(0,1.2)

#####figXXX = go.Figure(go.Bar(x=hist_x_data, y=hist_y_data))

#x, y = np.meshgrid(np.linspace(0, 10, 200), np.linspace(0, 10))
#y += 2 * np.sin(x * 2 * np.pi / 10)
#z = np.exp(-x / 10)

#figXXX = go.Figure(go.Scatter(x=x.flatten(), y=y.flatten(), mode="markers", marker_color=z.flatten()))


# figXXX = go.Figure(go.Histogram2d(
#         x=x,
#         y=y,
#         z=z
#     ))

#figXXX = px.imshow(z, text_auto=True, aspect="auto")


xx = x[0]
yy = y[:,0]

#THIS WORKS:
# # # # print(z.shape)
# # # # xx = x[0]
# # # # yy = y[:,0]
# # # # print(xx)
# # # # print(yy)
# # # # print(xx.shape)
# # # # print(yy.shape)
# # # # #sys.exit()
# # # # figXXX = px.imshow(z,
# # # #                 labels=dict(x="Lepton Hadron Angle", y="-t (GeV^2)", color="Ratio"),
# # # #                 x=xx,
# # # #                 y=yy
# # # #                )

globalblue = '#1670b5'

print(z)
figXXX = go.Figure(data=go.Heatmap(
                   z=z,
                   x=xx,
                   y=yy,
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
                '<b>Acc_Corr:  %{customdata[13]:.2f} +/- %{customdata[14]:.2f}<br>'+
                '<b>CLAS6 Result:  %{customdata[7]:.2f} +/- %{customdata[8]:.2f}<br>'+
                '<b>N_10.6GeV_sim  %{customdata[9]:.2f} ,N_5.75GeV_sim %{customdata[10]:.2f}<br>'+
                '<b>10.6GeV:5.75GeV Cross Section Ratio:  %{customdata[11]:.2f} +/- %{customdata[12]:.2f}<br>'+
                


                
                '<b>Percent of Lifters Stronger Than This</b>: %{text:.1f}%   <extra></extra>',
                #'test string {}'.format(x),
                text = ["this is text"],
                hoverlabel = {
                'bgcolor': globalblue,
                'font': {'color':'white'}
                }
                )
        #'test string {}'.format(x),



# pio.renderers.default = 'iframe'

# trace = go.Heatmap(
#     z=[[5, 7, 11], [8, 10, 12]],
#     colorbar = dict(title='Range'),
    
    
# )
# data = [trace]

# layout = go.Layout(xaxis=go.layout.XAxis(
#     title=go.layout.xaxis.Title(
#         text='Depth Axis',
#     )),
# yaxis=go.layout.YAxis(
#     title=go.layout.yaxis.Title(
#         text='Time Axis',
#     )
# ))
# f6 = go.Figure(data, layout=layout)
# pio.show(f6)

# labels=dict(x="Day of Week", y="Time of Day", color="Productivity"),
#                 x=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
#                 y=['Morning', 'Afternoon', 'Evening']
#                )

#fig.update_xaxes(side="top")
#fig.show()


# # # #print(np.dstack((arr1,arr2)))

# # # figXXX.update_traces(customdata=arr1,
# # #         hovertemplate = '<b>Weight Lifted in kg: (\u00B1 5)</b>: %{x:.0f} kg <br>'+\
# # #         '<b>Weight Lifted in lbs:</b>: %{customdata:.0f} lbs <br>'
# # #         #'test string {}'.format(x),
# # #         )

# # # newdats = np.stack((arr1,arr2),axis=-1)

# # # figXXX.update_traces(customdata=newdats,
# # #         hovertemplate = "<b> Point props<br>"+\
# # #                    "x: %{x}<br>"+\
# # #                    "y: %{y}<br>"+\
# # #                    "attr1: %{customdata[0]: .2f}<br>"+\
# # #                    "attr2: %{customdata[1]: .3f}")
# # #         #'test string {}'.format(x),




# # # """
# # # figXXX = go.Figure(go.Scatter(x=np.random.randint(3,9, 7),
# # #                  y=np.random.randint(-8, -3, 7),
# # #                  mode='markers',
# # #                  marker_size=18))




# # # num_attr =[2.4, 6.12, 4.5, 2.358, 8.23, 5.431, 7.4]

# # # figXXX.update_traces(customdata=num_attr,
# # #                    hovertemplate = "<b> Point prop<br>"+\
# # #                    "x: %{x}<br>"+\
# # #                    "y: %{y}<br>"+\
# # #                    "attr: %{customdata: .2f}");
# # # figXXX.update_layout(title='Scatterplot with hovertemplate from customdata with a single array')


# # # new_customdata = np.stack((num_attr, 100*np.random.rand(7)), axis=-1)
# # # new_customdata


# # # figXXX.update_traces(customdata = new_customdata,
# # #                     hovertemplate = "<b> Point props<br>"+\
# # #                    "x: %{x}<br>"+\
# # #                    "y: %{y}<br>"+\
# # #                    "attr1: %{customdata[0]: .2f}<br>"+\
# # #                    "attr2: %{customdata[1]: .3f}")
# # # figXXX.update_layout(title='Scatterplot with hovertemplate from   customdata of a two arrays')
# # # """
# # #     #print(sex_val)


# # # #print(year_value)
# # # #print(lift_val)

# # # #lift_set = lifts[Lift_Values.index(lift_val)]

# # # #print(lift_set)






outprint("Plotting Data")


app.layout = html.Div([
    html.Div([

            html.Div([
                dcc.RadioItems(
                    id='sex-type',
                    options=[{'label': i, 'value': i} for i in Sex_Values],
                    value=Sex_Values[1],
                    labelStyle={'display': 'inline-block'}
                )
            ],
            style={'width': '20%', 'padding-left':'10%', 'display': 'inline-block'}),


            html.Div([
                dcc.RadioItems(
                    id='lift-type',
                    options=[{'label': i, 'value': i} for i in Lift_Values],
                    value=Lift_Values[2],
                    labelStyle={'display': 'inline-block'}
                )
            ],
            style={'width': '30%', 'display': 'inline-block'}),

            html.Div([
                dcc.RadioItems(
                    id='equip-type',
                    options=[{'label': i, 'value': i} for i in equip_types],
                    value=equip_types[0],
                    labelStyle={'display': 'inline-block'}
                )
            ],
            style={'width': '40%', 'display': 'inline-block'}),



    dcc.Graph(id='indicator-graphic',figure=figXXX, config={'displayModeBar':False}),

    html.Label('Weight Class: Male', id='male-weight-class'),


    html.Div(
        html.Div(
            dcc.Slider(
                id='male-year--slider',
                min=min(wclasses[0]),
                max=max(wclasses[0]),
                value=wclasses[0][wclasses[0].index(min(wclasses[0]))+2],
                marks={
                    59 : '59 kg / 130 lb ',
                    66 : '66 kg / 145 lb',
                    74 : '74 kg / 163 lb',
                    83 : '83 kg / 183 lb',
                    93 : '93 kg / 205 lb',
                    105 : '105 kg / 231 lb',
                    120 : '120 kg / 265 lb',
                    128 : '120+',
                },
                step=None,
                updatemode='drag',
            ),
            id='malesliderContainer'),
        style={'width': '80%','padding-left':'10%', 'padding-right':'10%'}

        ),


    html.Label('Weight Class: Female', id='female-weight-class'),
    html.Div(
        html.Div(
            dcc.Slider(
                id='female-year--slider',
                min=min(wclasses[1]),
                max=max(wclasses[1]),
                value=wclasses[1][wclasses[1].index(min(wclasses[1]))+2],
                marks={
                        47 : '47 kg / 104 lb ',
                        52 : '52 kg / 115 lb',
                        57 : '57 kg / 126 lb',
                        63 : '63 kg / 139 lb',
                        72 : '72 kg / 159 lb',
                        84 : '84 kg / 185 lb',
                        90 : '84+',
                },
                step=None,
                updatemode='drag',
            ),
            id='femalesliderContainer'),
        style={'width': '80%','padding-left':'10%', 'padding-right':'10%'}

        )
])
])



# # # @app.callback(
# # #     Output('indicator-graphic', 'figure'),
# # #     [Input('sex-type', 'value'),
# # #     Input('lift-type', 'value'),
# # #     Input('equip-type', 'value'),
# # #     Input('indicator-graphic', 'hoverData'),
# # #     Input('female-year--slider', 'value'),
# # #     Input('male-year--slider', 'value')])
# # # def update_graph(sex_val,lift_val,equip_val,hoverdata,fe_year_value,male_year_value):

# # #     globalblue = '#1670b5'
# # #     colors = [globalblue,]*100
# # #     if hoverdata:
# # #         x_val = hoverdata['points']
# # #         #print(type(x_val))
# # #         xx = x_val[0]
# # #         x_ind = xx['pointNumber']
# # #         colors[x_ind] = '#59e94a' #green
# # #         #colors[x_ind] = '#e9524a' #red
# # #     #print(sex_val)

# # #     if Sex_Values.index(sex_val)==0:
# # #         year_value = male_year_value
# # #     else:
# # #         year_value = fe_year_value

# # #     #print(year_value)
# # #     #print(lift_val)

# # #     #lift_set = lifts[Lift_Values.index(lift_val)]

# # #     #print(lift_set)

# # #     equip_ind = equip_types.index(equip_val)
# # #     sex_ind = Sex_Values.index(sex_val)
# # #     weight_ind = wclasses[Sex_Values.index(sex_val)].index(year_value)
# # #     lift_ind = Lift_Values.index(lift_val)
# # #     #hist_data = lift_set[Sex_Values.index(sex_val)][wclasses[Sex_Values.index(sex_val)].index(year_value)]


# # #     hist_x_data = hist_bins
# # #     hist_y_data = lifts_hist_data[lift_ind][sex_ind][weight_ind][equip_ind]
# # #     hist_hover_labels = lifts_labels[lift_ind][sex_ind][weight_ind][equip_ind]

# # #     hist_lb_data = [i*2.20462 for i in hist_x_data]
# # #     hist_lb_low_data =  [(i+11) for i in hist_lb_data]
# # #     hist_kg_high_data = [(i+5) for i in hist_x_data]
# # #     #print(hist_hover_labels)

# # #     arr1 = np.array(hist_lb_data)
# # #     arr2 = np.array(hist_lb_low_data)
# # #     arr3 = np.array(hist_kg_high_data)

# # #     newdataX = np.stack((arr1,arr2,arr3),axis=-1)


# # #     #print(type(arr2))
# # #     fig10 = {
# # #         'data': [go.Bar(x=hist_x_data, y=hist_y_data, customdata=newdataX, marker_color = colors,
# # #                 hovertemplate = '<b>Weight Lifted in kg:</b>: %{x:.0f} - %{customdata[2]:.0f} kg <br>'+
# # #                 '<b>Weight Lifted in lbs:</b>: %{customdata[0]:.0f} - %{customdata[1]:.0f} lbs <br>'+
# # #                 '<b>Number of Lifters in Weight Range: </b>: %{y:.0f}<br>'+
# # #                 '<b>Percent of Lifters Stronger Than This</b>: %{text:.1f}%   <extra></extra>',
# # #                 #'test string {}'.format(x),
# # #                 text = hist_hover_labels,
# # #                 hoverlabel = {
# # #                 'bgcolor': globalblue,
# # #                 'font': {'color':'white'}
# # #                 }
# # #                 )
# # #                        #hovertext=hist_hover_labels)
# # #             ],
# # #         'layout': dict(
# # #             xaxis={
# # #                 'range': [0,500],
# # #                 'title': 'Weight Lifted(kg)',
# # #                 'textfont' : dict(
# # #                     family="Georgia",
# # #                     size=360,
# # #                     #color="#7f7f7f"
# # #                 ),
# # #                 'tickvals' : [100,200,300,400],
# # #                 'ticktext' : ['100 kg / 220 lbs','200 kg / 441 lbs','300 kg / 661 lbs','400 kg / 882 lbs']
# # #             },
# # #             yaxis={
# # #                 'title': "Number of Lifters"
# # #             },

# # #             #hovertext=["Text A", "Text B", "Text C", "Text D", "Text E"],
# # #             #hoverinfo="text",
# # #             margin={'l': 100, 'b': 40, 't': 10, 'r': 20},
# # #             #hovermode='x unified',
# # #             bargap=0.1, # gap between bars of adjacent location coordinates
# # #             bargroupgap=0.1 # gap between bars of the same location coordinates
# # #         )
# # #     }
# # # #
# # # #    fig10.update_traces(customdata= [hist_lb_data,hist_lb_low_data],
# # # #
# # # #                hovertemplate = '<b>Weight Lifted in kg: (\u00B1 5)</b>: %{x:.0f} kg <br>'+
# # # #                '<b>Weight Lifted in lbs: (\u00B1 10)</b>: %{customdata[0]:.0f} lbs <br>'+
# # # #                '<b>Weight Lifted in lbs: (\u00B1 10)</b>: %{customdata[1]:.0f} lbs <br>'+
# # # #                '<b>Number of Lifters in Weight Range: </b>: %{y:.0f}<br>'+
# # # #                '<b>Percent of Lifters Stronger Than This</b>: %{text:.1f}%   <extra></extra>',
# # # #    )

# # #     return fig10




# # # """
# # # @app.callback(Output('indicator-graphic', 'layout'),
# # #     [Input('indicator-graphic', 'hoverData')])
# # # def update_slide(hoverdata):



# # #     x_val = hoverdata['points']
# # #     #print(type(x_val))
# # #     xx = x_val[0]
# # #     x_ind = xx['pointNumber']

# # #     colors = ['#1670b5',]*100
# # #     #colors[25] = '#59e94a' #green
# # #     colors[25] = '#e9524a' #red

# # #     marker_color = colors

# # #     return
# # # """
# # #     #print(xx_val)
# # #     #if ind == 0:
# # #     #    return {}
# # #     #else:
# # #     #    print(hoverdata)
# # #         #return {'display': 'none'}



# # # @app.callback(Output('malesliderContainer', 'style'),
# # #     [Input('sex-type', 'value')])
# # # def update_slide(sex_val):

# # #     ind = Sex_Values.index(sex_val)

# # #     if ind == 0:
# # #         return {}
# # #     else:
# # #         return {'display': 'none'}



# # # @app.callback(Output('femalesliderContainer', 'style'),
# # #     [Input('sex-type', 'value')])
# # # def update_slide(sex_val):

# # #     ind = Sex_Values.index(sex_val)

# # #     if ind == 1:
# # #         return {}
# # #     else:
# # #         return {'display': 'none'}

# # # @app.callback(Output('female-weight-class', 'style'),
# # #     [Input('sex-type', 'value')])
# # # def update_slide(sex_val):

# # #     ind = Sex_Values.index(sex_val)

# # #     if ind == 1:
# # #         return {'width': '15%','padding-left':'10%', 'display': 'inline-block'}
# # #     else:
# # #         return {'display': 'none'}

# # # @app.callback(Output('male-weight-class', 'style'),
# # #     [Input('sex-type', 'value')])
# # # def update_slide(sex_val):

# # #     ind = Sex_Values.index(sex_val)

# # #     if ind == 0:
# # #         return {'width': '15%','padding-left':'10%', 'display': 'inline-block'}
# # #     else:
# # #         return {'display': 'none'}

if __name__ == '__main__':
    app.run_server(debug=False)
