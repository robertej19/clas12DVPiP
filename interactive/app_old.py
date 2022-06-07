import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
#import cPickle
#import gZip

import plotly.graph_objects as go

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


sex_val = 'Male'
lift_val = 'Squat'
equip_val = 'Raw'
male_year_value = 66

if Sex_Values.index(sex_val)==0:
    year_value = male_year_value
else:
    year_value = fe_year_value

equip_ind = equip_types.index(equip_val)
sex_ind = Sex_Values.index(sex_val)
weight_ind = wclasses[Sex_Values.index(sex_val)].index(year_value)
lift_ind = Lift_Values.index(lift_val)
#hist_data = lift_set[Sex_Values.index(sex_val)][wclasses[Sex_Values.index(sex_val)].index(year_value)]


hist_x_data = hist_bins
hist_y_data = lifts_hist_data[lift_ind][sex_ind][weight_ind][equip_ind]
hist_hover_labels = lifts_labels[lift_ind][sex_ind][weight_ind][equip_ind]

hist_lb_data = [i*2.20462 for i in hist_x_data]
hist_lb_low_data =  [(i-11) for i in hist_lb_data]
#print(hist_hover_labels)

arr1 = np.array(hist_lb_data)
arr2 = np.array(hist_lb_low_data)

figXXX = go.Figure(go.Bar(x=hist_x_data, y=hist_y_data))

#print(np.dstack((arr1,arr2)))

figXXX.update_traces(customdata=arr1,
        hovertemplate = '<b>Weight Lifted in kg: (\u00B1 5)</b>: %{x:.0f} kg <br>'+\
        '<b>Weight Lifted in lbs:</b>: %{customdata:.0f} lbs <br>'
        #'test string {}'.format(x),
        )

newdats = np.stack((arr1,arr2),axis=-1)

figXXX.update_traces(customdata=newdats,
        hovertemplate = "<b> Point props<br>"+\
                   "x: %{x}<br>"+\
                   "y: %{y}<br>"+\
                   "attr1: %{customdata[0]: .2f}<br>"+\
                   "attr2: %{customdata[1]: .3f}")
        #'test string {}'.format(x),




"""
figXXX = go.Figure(go.Scatter(x=np.random.randint(3,9, 7),
                 y=np.random.randint(-8, -3, 7),
                 mode='markers',
                 marker_size=18))




num_attr =[2.4, 6.12, 4.5, 2.358, 8.23, 5.431, 7.4]

figXXX.update_traces(customdata=num_attr,
                   hovertemplate = "<b> Point prop<br>"+\
                   "x: %{x}<br>"+\
                   "y: %{y}<br>"+\
                   "attr: %{customdata: .2f}");
figXXX.update_layout(title='Scatterplot with hovertemplate from customdata with a single array')


new_customdata = np.stack((num_attr, 100*np.random.rand(7)), axis=-1)
new_customdata


figXXX.update_traces(customdata = new_customdata,
                    hovertemplate = "<b> Point props<br>"+\
                   "x: %{x}<br>"+\
                   "y: %{y}<br>"+\
                   "attr1: %{customdata[0]: .2f}<br>"+\
                   "attr2: %{customdata[1]: .3f}")
figXXX.update_layout(title='Scatterplot with hovertemplate from   customdata of a two arrays')
"""
    #print(sex_val)


#print(year_value)
#print(lift_val)

#lift_set = lifts[Lift_Values.index(lift_val)]

#print(lift_set)






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



@app.callback(
    Output('indicator-graphic', 'figure'),
    [Input('sex-type', 'value'),
    Input('lift-type', 'value'),
    Input('equip-type', 'value'),
    Input('indicator-graphic', 'hoverData'),
    Input('female-year--slider', 'value'),
    Input('male-year--slider', 'value')])
def update_graph(sex_val,lift_val,equip_val,hoverdata,fe_year_value,male_year_value):

    globalblue = '#1670b5'
    colors = [globalblue,]*100
    if hoverdata:
        x_val = hoverdata['points']
        #print(type(x_val))
        xx = x_val[0]
        x_ind = xx['pointNumber']
        colors[x_ind] = '#59e94a' #green
        #colors[x_ind] = '#e9524a' #red
    #print(sex_val)

    if Sex_Values.index(sex_val)==0:
        year_value = male_year_value
    else:
        year_value = fe_year_value

    #print(year_value)
    #print(lift_val)

    #lift_set = lifts[Lift_Values.index(lift_val)]

    #print(lift_set)

    equip_ind = equip_types.index(equip_val)
    sex_ind = Sex_Values.index(sex_val)
    weight_ind = wclasses[Sex_Values.index(sex_val)].index(year_value)
    lift_ind = Lift_Values.index(lift_val)
    #hist_data = lift_set[Sex_Values.index(sex_val)][wclasses[Sex_Values.index(sex_val)].index(year_value)]


    hist_x_data = hist_bins
    hist_y_data = lifts_hist_data[lift_ind][sex_ind][weight_ind][equip_ind]
    hist_hover_labels = lifts_labels[lift_ind][sex_ind][weight_ind][equip_ind]

    hist_lb_data = [i*2.20462 for i in hist_x_data]
    hist_lb_low_data =  [(i+11) for i in hist_lb_data]
    hist_kg_high_data = [(i+5) for i in hist_x_data]
    #print(hist_hover_labels)

    arr1 = np.array(hist_lb_data)
    arr2 = np.array(hist_lb_low_data)
    arr3 = np.array(hist_kg_high_data)

    newdataX = np.stack((arr1,arr2,arr3),axis=-1)


    #print(type(arr2))
    fig10 = {
        'data': [go.Bar(x=hist_x_data, y=hist_y_data, customdata=newdataX, marker_color = colors,
                hovertemplate = '<b>Weight Lifted in kg:</b>: %{x:.0f} - %{customdata[2]:.0f} kg <br>'+
                '<b>Weight Lifted in lbs:</b>: %{customdata[0]:.0f} - %{customdata[1]:.0f} lbs <br>'+
                '<b>Number of Lifters in Weight Range: </b>: %{y:.0f}<br>'+
                '<b>Percent of Lifters Stronger Than This</b>: %{text:.1f}%   <extra></extra>',
                #'test string {}'.format(x),
                text = hist_hover_labels,
                hoverlabel = {
                'bgcolor': globalblue,
                'font': {'color':'white'}
                }
                )
                       #hovertext=hist_hover_labels)
            ],
        'layout': dict(
            xaxis={
                'range': [0,500],
                'title': 'Weight Lifted(kg)',
                'textfont' : dict(
                    family="Georgia",
                    size=360,
                    #color="#7f7f7f"
                ),
                'tickvals' : [100,200,300,400],
                'ticktext' : ['100 kg / 220 lbs','200 kg / 441 lbs','300 kg / 661 lbs','400 kg / 882 lbs']
            },
            yaxis={
                'title': "Number of Lifters"
            },

            #hovertext=["Text A", "Text B", "Text C", "Text D", "Text E"],
            #hoverinfo="text",
            margin={'l': 100, 'b': 40, 't': 10, 'r': 20},
            #hovermode='x unified',
            bargap=0.1, # gap between bars of adjacent location coordinates
            bargroupgap=0.1 # gap between bars of the same location coordinates
        )
    }
#
#    fig10.update_traces(customdata= [hist_lb_data,hist_lb_low_data],
#
#                hovertemplate = '<b>Weight Lifted in kg: (\u00B1 5)</b>: %{x:.0f} kg <br>'+
#                '<b>Weight Lifted in lbs: (\u00B1 10)</b>: %{customdata[0]:.0f} lbs <br>'+
#                '<b>Weight Lifted in lbs: (\u00B1 10)</b>: %{customdata[1]:.0f} lbs <br>'+
#                '<b>Number of Lifters in Weight Range: </b>: %{y:.0f}<br>'+
#                '<b>Percent of Lifters Stronger Than This</b>: %{text:.1f}%   <extra></extra>',
#    )

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



@app.callback(Output('malesliderContainer', 'style'),
    [Input('sex-type', 'value')])
def update_slide(sex_val):

    ind = Sex_Values.index(sex_val)

    if ind == 0:
        return {}
    else:
        return {'display': 'none'}



@app.callback(Output('femalesliderContainer', 'style'),
    [Input('sex-type', 'value')])
def update_slide(sex_val):

    ind = Sex_Values.index(sex_val)

    if ind == 1:
        return {}
    else:
        return {'display': 'none'}

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
