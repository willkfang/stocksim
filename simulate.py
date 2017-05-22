#!/bin/bash/

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
import datetime
import math

class Period():
#Data that holds information over a period
#Dates, Open, High, Low, Close, Volume, Adj Close
	def __init__(self,inputrow):
		self.row = inputrow
		self.refresh_cols()
	def refresh_cols(self):
		row = self.row
		self.date = row[0]
		self.open = row[1]
		self.high = row[2]
		self.low = row[3]
		self.close = row[4]
		self.adjclose = row[5]
		self.volume = row[6]


	def swap_cols(self, pos1, pos2):

		self.row[pos1], self.row[pos2] = self.row[pos2], self.row[pos1]

		self.refresh_cols()
	def return_close(self):
		return float(self.close)
	def return_open(self):
		return float(self.open)
	def return_volume(self):
		return float(self.volume)
	def return_date(self):
		return self.date
	def return_row(self):
		return self.row
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

		self.tickername = tickerfile

		with open(tickerfile,'r') as csvfile:
			file = csv.reader(csvfile)
			for row in file:
				# print ','.join(row)
				if "Date" not in row: #Remove Header
					self.daylist.append(Day(row))
				else:
					header = row
		# print header
		# for i in self.daylist:
		# 	i.print_date()
		if self.daylist[1].return_date() > self.daylist[2].return_date():
			self.daylist.reverse() #flip to chronological order
		self.ndays = len(self.daylist)

		head_numlist = []
		headlist = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
		for cols in header:
			num = -9
			for i in range(0, len(headlist)):
				if headlist[i][0] in cols[0]:
					head_numlist.append(i)

		# print head_numlist

		# print self.daylist[0].row
		
		for i in range(0,len(head_numlist)-1):
			if head_numlist[i] > head_numlist[i+1]:
				self.swapcols(self.daylist, i, i+1)

		# print self.daylist[0].row
		# print self.ndays

	def swapcols(self,thelist, pos1, pos2):
		for item in thelist:
			item.swap_cols(pos1,pos2)




	def calc_SMA(self, period):
		#Calculate simple moving average
		moving_avg = []
		# IV = self.daylist[0].return_close() #First close value
		period = int(round(period,0))
		for i in range(0, self.ndays):
			if i < period:
				IV = self.daylist[i].return_close() #First close value
				moving_avg.append(IV)
			else:
				total = 0
				for j in range(0,period):
					index = i-j
					total = total + self.daylist[index].return_close()
				total = total/period
				moving_avg.append(total)
		return moving_avg

	def calc_OBV(self):
		OBV = [self.daylist[0].return_volume()]

		for i in range(1,self.ndays):
			if self.daylist[i-1].return_close() < self.daylist[i].return_close():
				OBV.append(OBV[i-1] + self.daylist[i].return_volume())
			elif self.daylist[i-1].return_close() > self.daylist[i].return_close():
				OBV.append(OBV[i-1] - self.daylist[i].return_volume())
			else:
				OBV.append(OBV[i-1])

		return OBV
	def calc_rsi(self,period):
		RSI = [50]
		period = int(round(period,0))
		if period == 0:
			period = 1

		p_gain = 0
		p_loss = 0
		for i in range(1,self.ndays):

			dx = self.daylist[i].return_close() - self.daylist[i-1].return_close()

			if dx > 0:
				gain = dx
				loss = 0
			else:
				gain = 0
				loss = -dx

			if i < period:
				gain_avg = (p_gain * (i-1) + gain)/ i
				loss_avg = (p_loss * (i-1) + loss)/ i
			else:
				gain_avg = (p_gain * (period - 1) + gain)/ period
				loss_avg = (p_loss * (period - 1) + loss)/ period

			p_gain = gain_avg
			p_loss = loss_avg

			if loss_avg == 0:
				loss_avg = .000001

			RS = gain_avg/loss_avg

			RSI_value = 100 - 100 / (1 + RS)

			RSI.append(RSI_value)

		return RSI


# class TransactionHistory():
# 	def __init__(self):
# 		self.dates = []
# 		self.cash = []
# 		self.

class Wallet():
	def __init__(self, Funds):
		self.Initfunds = Funds
		self.cash = Funds
		self.securities = 0
		self.transhist = []
		self.worth = 0
		#transhist:
		#Date | Wallet

	def return_worth(self, price):
		return self.cash + self.securities * price

	def return_stockworth(self,price):
		return self.securities * price

	def buy(self, cost, nshares, date):
		totalprice = cost * nshares
		if totalprice < self.cash:
			self.cash = self.cash - totalprice
			self.securities += nshares
			self.transhist.append([date, 1, cost, nshares, self.cash])
		else:

			if math.floor(self.cash/cost) > 0:
				self.buy(cost, math.floor(self.cash/cost)-1, date)

	def sell(self, cost, nshares, date):
		if self.securities - nshares >= 0:
			totalprice = cost * nshares
			self.cash = self.cash + totalprice
			self.securities -= nshares
			self.transhist.append([date, -1, cost, nshares, self.cash])

		else:
			self.sell(cost, self.securities, date)

	def print_wallet(self, price):
		print "Cash: %f nshares: %f Total Worth: %f" %(self.cash, self.securities, self.return_worth(price))
	def print_transhist(self):
		for i in self.transhist:
			print i

class Simulation():
	#Class that holds the simulation

	def __init__(self,Ticker):
		#Initializes simulation
		#IF = initial Funds
		self.ticker = Ticker
		# self.IF = IF
		# self.funds = IF
		# self.shares = 0

		self.simhistory = []

		self.tickername = self.ticker.tickername

	def calc_buyhold(self):


		Initial_funds = self.wallet.Initfunds
		first_price = self.ticker.daylist[0].return_close()
		current_price = self.ticker.daylist[-1].return_close()
		baseline = Initial_funds/first_price * current_price
		return baseline

	def buy(self,i,nshares, inpercent):
		cost = self.ticker.daylist[i].return_close()
		date = self.ticker.daylist[i].date
		if nshares < 0:
			nshares = 0
		if inpercent == True:
			nshares = math.floor(nshares * self.wallet.cash/cost)
		self.wallet.buy(cost, nshares,date)
		# if self.funds > cost*nshares:
		# 	self.funds = self.funds - cost*nshares
		# 	self.shares = self.shares + nshares

	def sell(self,i,nshares, inpercent):
		cost = self.ticker.daylist[i].return_close()
		date = self.ticker.daylist[i].date
		if nshares < 0:
			nshares = 0

		if inpercent == True:
			nshares = math.floor(nshares * self.wallet.cash/cost)

		self.wallet.sell(cost, nshares,date)


	def buy_criteria(self,coefficients, i):
		#Criteria to buy

		SMA1 = self.SMA1[i-1]
		SMA2 = self.SMA2[i-1]
		RSI1 = self.RSI1[i-1]
		dOBV = self.OBV[i-1]
		price = self.ticker.daylist[i-1].return_close()
		coef = coefficients

		crit = coef[0] * (price-SMA1) + coef[1] * (price-SMA2) + coef[2] * RSI1 + coef[3] * dOBV
		# print crit
		return crit

	def sell_criteria(self,coefficients, i):
		#Criteria to sell
		SMA1 = self.SMA1[i-1]
		SMA2 = self.SMA2[i-1]
		RSI1 = self.RSI1[i-1]
		dOBV = self.OBV[i-1]
		price = self.ticker.daylist[i-1].return_close()
		coef = coefficients

		crit = coef[0] * (price-SMA1) + coef[1] * (price-SMA2) + coef[2] * RSI1 + coef[3] * dOBV

		# print crit
		return crit

	def refresh(self):
		simhistory = []

	# def return_sma_value(self, SMA, date):
	# 	value  = 0


	# 	return value
	def calc_dx(self):
		self.dx = []
		for i in range(1, self.ticker.ndays):
			self.dx.append(self.ticker.daylist[i].return_close - self.ticker.daylist[i-1].return_close)


	def simulate(self, coefficients, IF):
		self.wallet = Wallet(IF)

		self.baseline = self.calc_buyhold()

		self.simhistory = []



		first_SMAcoef = 6
		nSMA = 4
		ncoef = len(coefficients) - nSMA - 1
		buycoef = coefficients[first_SMAcoef:(ncoef/2 + first_SMAcoef)]
		sellcoef = coefficients[(first_SMAcoef + ncoef/2 -1 ):len(coefficients)]

		buyfactor = coefficients[3]
		sellfactor = coefficients[4]

		#######Indicators
		#SMA
		self.SMA1 = self.ticker.calc_SMA(coefficients[0])
		self.SMA2 = self.ticker.calc_SMA(coefficients[1])

		#RSI
		self.RSI1 = self.ticker.calc_rsi(coefficients[2])

		#OBV
		self.OBV = self.ticker.calc_OBV()
		
		self.simhistory.append(['Date', 'todayprice', 'num_securities', 'Cash', 'Stockworth', 'Totalworth', 'Baseline Comparison'])
		
		for i in range(1,self.ticker.ndays):
			
			day = self.ticker.daylist[i].date
			todayprice = self.ticker.daylist[i].return_close()
			yestprice = self.ticker.daylist[i-1].return_close()
			output = "Held"
			if i-1 < 0:
				yestprice = self.ticker.daylist[0]

			buy_crit = self.buy_criteria(buycoef, i)
			sell_crit = self.sell_criteria(sellcoef,i)

			if buy_crit > 0:
				self.buy(i, buy_crit/ (buy_crit + buyfactor), True ) #second is num shares bought. Should vary with indicators
			
			if sell_crit > 0:
				self.sell(i, sell_crit / (sell_crit + sellfactor), True) 


			totalworth = self.wallet.return_worth(todayprice)
			# print "Date: %s, Today's price: %f, Yesterday's Price: %f, Wallet: %f, Shares: %f" %(day, todayprice, yestprice, totalworth, self.wallet.securities)
			# print self.SMA10[i]
			stdtime = time.mktime(datetime.datetime.strptime(day, "%m/%d/%Y").timetuple())
			# stdtime = datetime.datetime.strptime(day, "%m/%d/%Y").date()
			# stdtime = datetime.datetime.strptime(day, "%y/%m/%d").date()
			# self.simhistory.append([stdtime, todayprice, self.wallet.securities, self.wallet.cash, self.wallet.return_stockworth(todayprice), totalworth])
			self.simhistory.append([day, todayprice, self.wallet.securities, self.wallet.cash, self.wallet.return_stockworth(todayprice), totalworth, totalworth - self.baseline])
			# self.wallet.print_wallet(todayprice)
		# print self.wallet.print_transhist()
		# print self.simhistory
		# self.simhistory = np.asarray(self.simhistory)
		# print self.simhistory

		return totalworth - self.baseline

	def plotsimulation(self):
		days = self.simhistory[:,0]
		days2 = []
		# for i in days:
		# 	# matplotlib.dates.date2num
		# 	days2.append(time.strftime('%Y-%m-%d', time.localtime(days[0])))


		price = self.simhistory[:,1]
		nsecurities = self.simhistory[:,2]
		cash = self.simhistory[:,3]
		stockworth = self.simhistory[:,4]
		totalworth = self.simhistory[:,5]
		# dates = matplotlib.dates.date2num(np.ndarray.tolist(days))

		plt.figure(2)   

		plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
		plt.gca().xaxis.set_major_locator(mdates.DayLocator())
		# plt.plot(days,price*4)
		plt.plot(days,price)
		plt.gcf().autofmt_xdate()

def readmatrix(fname):
	with open(fname,'r') as csvfile:
			file = csv.reader(csvfile)
			count = 0
			matrix = []
			for row in file:
				count = count+1
				# print ','.join(row)
				if count > 12: #Remove Header
					matrix.append(row)
	return matrix
def print_resultstofile(fname,array):
	f = open(fname, 'w')
	for line in array:
		printline = str(line).replace("[","").replace("]","")
		# print printline
		f.write(printline)
		f.write('\n')
def convert2float(inputarray):
	array2 = []
	for i in inputarray:
		array2.append(float(i))
	return array2

class Tickerlist():
	def __init__(self,tickers):
		self.listoftickers = []
		self.listoftickersims = []
		self.ntickers = len(tickers)

		for i in range(0,self.ntickers):
			self.listoftickers.append(Ticker(tickers[i]))
			self.listoftickersims.append(Simulation(self.listoftickers[i]))

	def return_avg_sim_all(self, coefficients,Initial_Funding, printoption):
		results = []
		for tickersim in self.listoftickersims:
			result = tickersim.simulate(coefficients, Initial_Funding)
			results.append(result)
			if printoption == True:
				printfilename = 'history_' + tickersim.tickername
				print_resultstofile(printfilename, tickersim.simhistory)
				print "{0}: {1}".format(tickersim.tickername, result)


		return return_avg(results)
	
	def refresh_all(self):
		for tickersims in self.listoftickersims:
			tickersims.refresh()

def return_avg(thelist):
	total = 0
	for nums in thelist:
		if nums < 0:
			nums = 5*nums
		total = total + float(nums)
	return total/len(thelist)


def main():
	coefficients = [78.03225806, 187.1612903, 65.19354839, 93.5483871, 9.677419355, -677.4193548, -32.25806452, -935.483871, -32.25806452, 1000.0, 161.2903226, -612.9032258, -806.4516129]
	coefmatrix = readmatrix('Oarray_vix2.csv')
	doe = 1 #DOE or use coefficients
	resultsmatrix = []

	# Ticker1 = Ticker('VB.csv')
	# SPY = Ticker('AMD.csv')
	# VXX = Ticker('VXX.csv')

	# SPY_5daySMA = SPY.calc_SMA(5)
	# SPY_10daySMA = SPY.calc_SMA(10)
	# plt.plot(SPY_5daySMA)
	# plt.plot(SPY_10daySMA)
	# Ticker1Simulation = Simulation(Ticker1)
	# SPYSimulation = Simulation(SPY)
	# VXXSimulation = Simulation(VXX)
	
	output = []

	Initial_Funding = 10000

	tickers = ['VB.csv', 'SPY.csv', 'AMD.csv', 'F.csv']
	tickerlist = Tickerlist(tickers)

	if doe == 1:
		for i in range(0,len(coefmatrix)):
			coefficients = convert2float(coefmatrix[i][1:len(coefmatrix[i])])
			# Ticker1result = Ticker1Simulation.simulate(coefficients, Initial_Funding)
			# SPYresult = SPYSimulation.simulate(coefficients, Initial_Funding)
			# VXXresult = VXXSimulation.simulate(coefficients, Initial_Funding)
			# print coefficients
			# avg_result = (Ticker1result + SPYresult)
			avg_result = tickerlist.return_avg_sim_all(coefficients,Initial_Funding, False)
			output.append([coefficients, avg_result])

			tickerlist.refresh_all()
			# Ticker1Simulation.refresh()
			# SPYSimulation.refresh()
			# VXXSimulation.refresh()

			if i%100 == 0:
				print i
		
		sortedoutput = sorted(output, key=lambda x : x[-1])
		print sortedoutput[-1]
		print sortedoutput[-2]
		print sortedoutput[-3]
		print_resultstofile('output_all.csv', sortedoutput)

		coefficients = sortedoutput[-1][0]
		print coefficients

		tickerlist.return_avg_sim_all(coefficients,Initial_Funding, True)

		# Ticker1_result = Ticker1Simulation.simulate(coefficients, 10000)
		# print_resultstofile('history_Ticker1.csv', Ticker1Simulation.simhistory)
		# print "Ticker1: %f" % Ticker1_result

		# SPY_result = SPYSimulation.simulate(coefficients, 10000)
		# print_resultstofile('history_SPY.csv', SPYSimulation.simhistory)

		# print "SPY: %f" % SPY_result


	else:	
		tickerlist.return_avg_sim_all(coefficients,Initial_Funding, True)
		# Ticker1_result = Ticker1Simulation.simulate(coefficients, 10000)
		# print_resultstofile('history_Ticker1.csv', Ticker1Simulation.simhistory)
		# print "Ticker1: %f" % Ticker1_result

		# SPY_result = SPYSimulation.simulate(coefficients, 10000)
		# print_resultstofile('history_SPY.csv', SPYSimulation.simhistory)

		# print "SPY: %f" % SPY_result

		# VXX_result = VXXSimulation.simulate(coefficients, 10000)
		# print_resultstofile('history_VXX.csv', VXXSimulation.simhistory)	
		# print "VXX: %f" % VXX_result

	# SPYSimulation.plotsimulation()

	# plt.show()
if __name__ == '__main__':
	main()