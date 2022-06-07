import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os, subprocess
import math
import shutil
from icecream import ic


def plot_2dhist(x_data,y_data,var_names,ranges,colorbar=True,
            saveplot=False,pics_dir="none",plot_title="none",logger=False,first_label="rad",
            filename="ExamplePlot",units=["",""],extra_data=None):
    
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "30"
    # Initalize parameters
    x_name = var_names[0]
    y_name = var_names[1]
    xmin = ranges[0][0]
    xmax =  ranges[0][1]
    num_xbins = ranges[0][2]
    ymin = ranges[1][0]
    ymax =  ranges[1][1]
    num_ybins = ranges[1][2]
    x_bins = np.linspace(xmin, xmax, num_xbins) 
    y_bins = np.linspace(ymin, ymax, num_ybins) 

    # Creating plot
    fig, ax = plt.subplots(figsize =(18, 10)) 
    if units[0] == "None":
        ax.set_xlabel("{}".format(x_name))
    else:
        ax.set_xlabel("{} ({})".format(x_name,units[0]))
    if units[1] == "None":
        ax.set_ylabel("{}".format(y_name))
    else:
        ax.set_ylabel("{} ({})".format(y_name,units[1]))

    plt.hist2d(x_data, y_data, bins =[x_bins, y_bins],
        range=[[xmin,xmax],[ymin,ymax]],norm=mpl.colors.LogNorm())# cmap = plt.cm.nipy_spectral) 

    # Adding color bar 
    if colorbar:
        plt.colorbar()

    #plt.tight_layout()  

    
    #Generate plot title
    if plot_title == "none":
        plot_title = '{} vs {}'.format(x_name,y_name)
    
    plt.title(plot_title) 
        
    
    if saveplot:
        #plot_title.replace("/","")
        new_plot_title = plot_title.replace("/","").replace(" ","_").replace("$","").replace("^","").replace("\\","").replace(".","").replace("<","").replace(">","")
        print(new_plot_title)
        if not os.path.exists(pics_dir):
            os.makedirs(pics_dir)
        plt.savefig(pics_dir + new_plot_title+".png")
        plt.close()
        print("Figure {} saved to {}".format(new_plot_title,pics_dir))

    else:
       plt.show()


    if extra_data is not None:
        plt.hist2d(extra_data[0], extra_data[1], bins =[x_bins, y_bins],
            range=[[xmin,xmax],[ymin,ymax]], cmap = plt.cm.nipy_spectral) #norm=mpl.colors.LogNorm())#
        plt.show()

def plot_1dhist(x_data,vars,ranges="none",second_x=False,second_x_data=[],logger=False,first_label="rad",second_label="norad",
            saveplot=False,pics_dir="none",plot_title="none",first_color="blue",sci_on=False,plot_title_identifiyer="",addvars=None):
    
    if second_x:
        if len(x_data)<len(second_x_data):
            second_x_data = second_x_data.sample(n=len(x_data))
        elif len(x_data)>len(second_x_data):
            x_data = x_data.sample(n=len(second_x_data))
        

    if x_data.dtype == "float64":
        plot_title = plot_title
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = "20"

        # Initalize parameters
        x_name = vars[0]

        if ranges=="none":
            xmin = 0.99*min(x_data)
            xmax =  1.01*max(x_data)
            num_xbins = 50#int(len(x_data)/)
        else:
            xmin = ranges[0]
            xmax =  ranges[1]
            num_xbins = ranges[2]

        x_bins = np.linspace(xmin, xmax, num_xbins) 

        # Creating plot
        fig, ax = plt.subplots(figsize =(18, 10)) 
            
        ax.set_xlabel(x_name)  
        ax.set_ylabel('counts')  
        
        a = first_label
        a2 = second_label
        b = "rad"
        b2="norad"
        plt.hist(x_data, bins =x_bins, range=[xmin,xmax], color='blue', alpha=0.5, label=a)# cmap = plt.cm.nipy_spectral) 
        if second_x:
            plt.hist(second_x_data, bins =x_bins, range=[xmin,xmax],color='red', alpha=0.5, label=a2)# cmap = plt.cm.nipy_spectral) 
            plt.legend()


        #plt.plot(addvars[1], addvars[0])

        from scipy.stats import norm


        plt.hist(x_data, density=False)
        #plt.xlim((min(arr), max(arr)))

        mu = np.mean(x_data)
        variance = np.var(x_data)
        sigma = np.sqrt(variance)
        x = np.linspace(min(x_data), max(x_data), 100)
        plt.plot(x,  norm.pdf(x, mu, sigma))

        
        #plt.tight_layout()  

        if logger:
            plt.yscale('log')

        #Generate plot title
        if plot_title == "none":
            plot_title = '{} counts'.format(x_name)
        
        plt.title(plot_title) 
        
        if sci_on:
            plt.ticklabel_format(axis="x",style="sci",scilimits=(0,0))


        if saveplot:
            new_plot_title = plot_title.replace("/","").replace(" ","_").replace("$","").replace("^","").replace("\\","").replace(".","").replace("<","").replace(">","")
            #print(new_plot_title)
            

            #print(pics_dir)
            if not os.path.exists(pics_dir):
                os.makedirs(pics_dir)

            plt.savefig(pics_dir + plot_title_identifiyer+new_plot_title+".png")
            #plt.savefig(pics_dir + new_plot_title+"_linear_scale"+".png")

            plt.close()
        else:
            plt.show()
    #except OSError as error: 
    #    print(error)  





if __name__ == "__main__":
    ranges = [0,1,100,0,300,120]
    variables = ['xB','Phi']
    conditions = "none"
    datafile = "F18In_168_20210129/skims-168.pkl"
    plot_2dhist(datafile,variables,ranges)