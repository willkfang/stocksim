#!/bin/bash/

import csv
import numpy as np
import matplotlib.pyplot as plt

class Period():
#Data that holds information over a period
#Dates, Open, High, Low, Close, Volume, Adj Close
	def __init__(self,row):
		self.date = row[0]
		self.open = row[1]
		self.high = row[2]
		self.low = row[3]
		self.close = row[4]
		self.volume = row[5]
		self.adjclose = row[6]
		self.row = row
	def return_close(self):
		return float(self.close)

class Day(Period):
#Specific day period
	
	def __init__(self,row):
		Period.__init__(self,row)

	def print_date(self):
		print self.date

class Ticker():
#Datasettype that contains and holds the historical ticker information
	def __init__(self,tickerfile):
		#Create datatype ticker, read in file and assign to lists
		self.daylist = []
		self.datelist = []
		self.openlist = []
		self.highlist = []
		self.lowlist = []
		self.closelist = []
		self.volume = []
		self.adjclose = []

		with open(tickerfile,'r') as csvfile:
			file = csv.reader(csvfile)
			for row in file:
				# print ','.join(row)
				if "Date" not in row:
					self.daylist.append(Day(row))

		# for i in self.daylist:
		# 	i.print_date()
	
		self.ndays = len(self.daylist)
		# print self.ndays

	def calc_SMA(self, period):
		moving_avg = []

		for i in range(0, self.ndays):
			if i < period:
				moving_avg.append(0)
			else:
				total = 0
				for j in range(0,period):
					index = i-j
					total = total + self.daylist[index].return_close()
				moving_avg.append(total)
		return moving_avg 

def main():
	SPY = Ticker('spy1.csv')
	SPY_5daySMA = SPY.calc_SMA(5)

	plt.plot(SPY_5daySMA)

	x = np.arange(0, 5, 0.1);
	y = np.sin(x)
	plt.plot(x, y)

if __name__ == '__main__':
	main()