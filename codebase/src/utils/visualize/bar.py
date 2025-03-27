""" This script contains functions to draw horizontal "bar"-type
plots in a given axis within a figure, e.g. gene location lines,
linplot of signal/intensity vector.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scienceplots

""" Function to draw boxes corresponding to genes
along a horizontal line
@param ax: axis/subplot to draw the line in
@param genes: list of genes returned from readdata.processGeneFile()
@param n: number of bins/regions (i.e. row dimension of HiC matrix) 
"""
plt.style.use('science')

def drawGenes(ax,genes,n):
	plt.style.use('science')
	ax.set_xlim([0,n])
	ax.set_ylim([-1,1])
	ax.tick_params(which='both',
				right=False,left=False,top=False,bottom=False,
				labelbottom=False,labelleft=False)
	cmap=plt.get_cmap("tab20c")
	genes = genes.iloc[::-1]
	for idx, row in genes.iterrows():
		start = row['start']
		length = row['end']-row['start']
		rect = patches.Rectangle((start,-1),length,2,facecolor=cmap(idx%20),edgecolor=None) #cmap(idx%20))
		# rect = patches.Rectangle((start,-1),length,2,edgecolor=None) #cmap(idx%20))
		ax.add_patch(rect)
	ax.plot([0,n],[0,0],'k-',linewidth=1,zorder=-1)

""" Function to draw a filled-in line plot of some signal or
intensity vector
@param ax: axis/subplot to draw the line in
@param seqDepth: numpy 1D array/vector containing the signal/intensity
@param color: color to fill the space between the line and x-axis
"""
def drawTrack(ax, seqDepth, color,  ylim):
	n = seqDepth.shape[0]
	ax.set_ylim([0,ylim])
	ax.set_xlim([0,n])
	ax.xaxis.set_ticks(np.arange(0, n, n/4))
	ax.tick_params(which='both', direction='out',
			right=False,left=False,top=False,
			labelbottom=False,labelleft=False,
			length=4)
	ax.fill_between(range(n),0,seqDepth,color=color)

def drawTrack2(ax, seqDepth, color,ymin,ylim):
	n = seqDepth.shape[0]
	ax.set_ylim([ymin,ylim])
	ax.set_xlim([0,n])
	ax.xaxis.set_ticks(np.arange(0, n, n/4))
	ax.tick_params(which='both', direction='out',
			right=False,left=True,top=False,
			labelbottom=False,labelleft=True,
			length=4)
	ax.fill_between(range(n),0,seqDepth,color=color)

def drawMarks(ax, n, seqDepth, color, marker):
	ax.set_xlim([0,n])
	x = seqDepth 
	y = np.zeros(x.shape)
	ax.xaxis.set_ticks(np.arange(0, n, n/4))
	ax.tick_params(which='both', direction='out',
			right=False,left=False,top=False,
			labelbottom=False,labelleft=False,
			length=4)
	ax.plot(x, y, marker=marker,color=color, linestyle='None',markersize=6)
