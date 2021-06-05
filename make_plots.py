from os import listdir
from os.path import isfile, join
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import gridspec, rc
import math

font = {'size'   : 17}

rc('font', **font)

def plot_figures(dataset_name, ax):
	onlyfiles = [f for f in listdir(".") if isfile(join(".", f))]
	colors = {
		"DLAL": "black",
		"EAL": "blue",
		"MVAL": "red",
		"RS": "slategrey",
		"RCAL": "green",
		"LBAL": "yellow",
		"EGLAL": "pink",
		"LCAL": "orange",
		"MCAL": "purple",
		"EIDAL": "lightskyblue",
	}
	counter = 0
	for file in onlyfiles:
		if not ("pkl" in file and dataset_name in file):
			continue
		alName = file.split("_")[0]
		if alName not in colors:
			continue
		counter += 1
		alDict = pickle.load(open(file, 'rb'))
		scores = [val/alDict["max_score"] if val <= alDict["max_score"] else 1.0 for val in alDict["scores"]]
		needed_points = alDict["num_points"][-1]
		if alName == "DLAL":
			linestyle = "solid"
			linewidth = 2
			z_order = 1000
		else:
			linestyle = "dotted"
			linewidth = 1
			z_order = counter
			if alName == "RS":
				z_order = 900
		ax.plot(alDict["num_points"], scores, label=alName, color=colors[alName], linestyle = linestyle, linewidth = linewidth, zorder=z_order)
	ax.set_title(dataset_name, y=-0.01)
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
data_lst = ["Digits", "Iris", "Wine", "Balance", "Car", "Pen", "Mushroom", "Heart", "SatImage", "Glass", "Segmentation", "Vowel", "Dermatology", "IoT"]

gs = gridspec.GridSpec(4, math.ceil(len(data_lst)/4.0))
fig = plt.figure(figsize=(50, 20))

gridspaces = [i for i in range(4*(len(data_lst)//4))]
gridspaces.extend([gridspaces[-1] + 2, gridspaces[-1] + 3])
print(len(gridspaces))
print(len(data_lst))
counter = 0
for n in gridspaces:
	ax = fig.add_subplot(gs[n])
	data = data_lst[counter]
	plot_figures(data, ax)
	handles, labels = ax.get_legend_handles_labels()
	counter += 1

fig.legend(handles, labels, ncol=5, loc='lower center')
#fig.legend(handles, labels, 'lower right')
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)
plt.savefig('Al_plots.png', bbox_inches='tight')
plt.savefig('Al_plots.pdf', bbox_inches='tight')