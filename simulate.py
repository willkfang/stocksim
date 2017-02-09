#!/bin/bash/

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
import datetime

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
	def return_date(self):
		return self.date

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
				if "Date" not in row: #Remove Header
					self.daylist.append(Day(row))

		# for i in self.daylist:
		# 	i.print_date()
		self.daylist.reverse() #flip to chronological order
		self.ndays = len(self.daylist)
		# print self.ndays

	def calc_SMA(self, period):
		#Calculate simple moving average
		moving_avg = []
		IV = self.daylist[0].return_close() #First close value
		for i in range(0, self.ndays):
			if i < period:
				moving_avg.append(IV)
			else:
				total = 0
				for j in range(0,period):
					index = i-j
					total = total + self.daylist[index].return_close()
				total = total/period
				moving_avg.append(total)
		return moving_avg

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

	def sell(self, cost, nshares, date):
		totalprice = cost * nshares
		self.cash = self.cash + totalprice
		self.securities -= nshares
		self.transhist.append([date, -1, cost, nshares, self.cash])


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

	def buy(self,i,nshares):
		cost = self.ticker.daylist[i].return_close()
		date = self.ticker.daylist[i].date
		self.wallet.buy(cost, nshares,date)
		# if self.funds > cost*nshares:
		# 	self.funds = self.funds - cost*nshares
		# 	self.shares = self.shares + nshares

	def sell(self,i,nshares):
		cost = self.ticker.daylist[i].return_close()
		date = self.ticker.daylist[i].date
		self.wallet.sell(cost, nshares,date)


	def buy_criteria(self,coefficients):
		#Criteria to buy

		return True

	def sell_criteria(self,coefficients):
		#Criteria to sell

		return True

	def simulate(self, coefficients, IF):
		self.wallet = Wallet(IF)

		for i in range(1,self.ticker.ndays):
			#Test strategy, buy if cost is lower than previous day

			day = self.ticker.daylist[i].date
			todayprice = self.ticker.daylist[i].return_close()
			yestprice = self.ticker.daylist[i-1].return_close()
			output = "Held"
			if i-1 < 0:
				yestprice = self.ticker.daylist[0]

			if self.buy_criteria(coefficients):
				self.buy(i,2)
			if self.sell_criteria(coefficients):
				self.sell(i,2)


			totalworth = self.wallet.return_worth(todayprice)
			print "Date: %s, Today's price: %f, Yesterday's Price: %f, Wallet: %f" %(day, todayprice, yestprice, totalworth)
			
			# stdtime = time.mktime(datetime.datetime.strptime(day, "%m/%d/%Y").timetuple())
			stdtime = datetime.datetime.strptime(day, "%m/%d/%Y").date()

			self.simhistory.append([stdtime, todayprice, self.wallet.securities, self.wallet.cash, self.wallet.return_stockworth(todayprice), totalworth])
			self.wallet.print_wallet(todayprice)
		# print self.wallet.print_transhist()
		# print self.simhistory
		self.simhistory = np.asarray(self.simhistory)
		# print self.simhistory
		return totalworth
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
		plt.plot(days,totalworth)
		plt.gcf().autofmt_xdate()

def main():
	coefficients = []
	SPY = Ticker('spy1.csv')
	SPY_5daySMA = SPY.calc_SMA(5)
	SPY_10daySMA = SPY.calc_SMA(10)
	# plt.plot(SPY_5daySMA)
	# plt.plot(SPY_10daySMA)
	SPYSimulation = Simulation(SPY)
	SPYSimulation.simulate(coefficients, 1000)
	SPYSimulation.plotsimulation()
	plt.show()
if __name__ == '__main__':
	main()