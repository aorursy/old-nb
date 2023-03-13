# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import numpy as np
import matplotlib.pyplot as plt

#FUNCTIONS
#--is_number
#--calc_entropy
#--calc_gain
#---------------------------------------------------------------------------------------------
def is_number(s):
	try:
		return float(s)
	except ValueError:
		return False

# Entropy(TotalSet) = - (Class_Yes/NumberInTotal)*Log2(Class_Yest/NumberInTotal) - (Class_No/NumberInTotal)*Log2(Class_No/NumberInTotal))
def calc_entropy(NumberInTotal,NumberTrueInTotal):	
	total = float(NumberInTotal)
	totalTrue = float(NumberTrueInTotal)
	totalFalse = total - totalTrue

	if total != 0:
		tF_div_t = totalFalse / total

		if totalTrue != 0:
			tT_div_t = totalTrue / total
		else:
			tT_div_t = 0
	else:
		tF_div_t = 0
		tT_div_t = 0

	if tT_div_t != 0:
		entropyTrue = -tT_div_t * np.log2(tT_div_t)
	else:
		entropyTrue = 0
	
	if tF_div_t != 0:
		entropyFalse = -tF_div_t * np.log2(tF_div_t)
	else:
		entropyFalse = 0

	#entropy
	return  entropyTrue + entropyFalse

# Gain = Entropy(TotalSet) - Sum ( ( NumberInSubset / NumberInTotal) ) * Entropy(Subset)
def calc_gain(EntropyOfTotalSet,NumberInTotal, ArrayOf_NumberInSubset,ArrayOf_NumberTrueInSubset):
	sumOfSubsetsEntropy = 0
	subsetIndex = 0
	numberOfSubsets = len(ArrayOf_NumberInSubset)
	while subsetIndex < numberOfSubsets:
		sumOfSubsetsEntropy += (ArrayOf_NumberInSubset[subsetIndex]/float(NumberInTotal)) * calc_entropy( ArrayOf_NumberInSubset[subsetIndex],ArrayOf_NumberTrueInSubset[subsetIndex])
	 	subsetIndex += 1
	return EntropyOfTotalSet - sumOfSubsetsEntropy #gain
#---------------------------------------------------------------------------------------------

#CLASSES
#---Node
#---Tree
#******************************************************************
class Node:
	def __init__(self,typeOfNode,nodeId):
		#types are 'branching' and 'value'
		#values can include 
			# a numerical percentage chance of membership
			# membership value of 1 or 0 (extremely confident)
			# failure
		#branching nodes lead to other nodes
		self._typeOfNode = typeOfNode
		self._ID = nodeId
		if typeOfNode == 'Branching':
			self._paths = []

	def setBranchType(self,branchType):
		self._branchType = branchType

	def setFeatureIndex(self,featureIndex):
		self._featureIndex = featureIndex

	def setValue(self,value):
		self._value = value

	def setAddress(self,address): # this is the nodes 'address' in the previous branch nodeIs' _paths list
		self._address = address

	def setSplitValue(self,splitValue):
		self._splitValue = splitValue

	def setGroupWithHighestOdds(self,groupWithHighestOdds):
		self._groupWithHighestOdds = groupWithHighestOdds

	def getBranchType(self):
		return self._branchType
	def getFeatureIndex(self):
		return self._featureIndex
	def getId(self):
		return self._ID
	def getType(self):
		return self._typeOfNode
	def getValue(self):
		return self._value
	def getAddress(self):
		return self._address
	def getSplitValue(self):
		return self._splitValue
	def getGroupWithHighestOdds(self):
		return self._groupWithHighestOdds

	# arg 'branch' is nodeID the node branches to
	def addBranch(self,branch):
		self._paths.append(branch)	
	def getPaths(self):
		return self._paths

class Tree:
	def __init__(self,featuresList,membershipData,featureData):
		self._nodes = {}
		self._nodeCount = 0
		
		numberTrue = 0
		numberFalse = 0
		unknownValue = 0
		for memberData in membershipData:
		 	if memberData == '1':
		 		numberTrue += 1
		 	elif memberData == '0':
		 		numberFalse += 1
		 	else:
		 		unknownValue += 1

		overAllOddsOfMembership = numberTrue / float(numberTrue+numberFalse)

		self._rootNode = self.CreateNode(featuresList,membershipData,featureData,overAllOddsOfMembership)

	def CreateNode(self,featuresList,membershipData,featureData,overAllOddsOfMembership):
		#*If all cases are true, create true node
		#**Get number of true false for current data
		print 'node creating'
		numberTrue = 0
		numberFalse = 0
		unknownValue = 0
		for memberData in membershipData:
		 	if memberData == '1':
		 		numberTrue += 1
		 	elif memberData == '0':
		 		numberFalse += 1
		 	else:
		 		unknownValue += 1

		#*If no data, create failure node
		if len(featureData) == 0:
			node = Node('value',str(self._nodeCount))
			self._nodeCount += 1
			print numberTrue
			print numberFalse
			node.setValue(overAllOddsOfMembership)
			
			self._nodes[node.getId()] = node
			##print 'FAILURE node'
			#raw_input()
			return node
		

		if numberFalse == 0 and numberTrue > 0:
		 	node = Node('value',str(self._nodeCount))
			self._nodeCount += 1
		 	node.setValue(1.0)
			self._nodes[node.getId()] = node
			##print 'true node'
			#raw_input()
			return node


		 #*If all cases are false, create false node
		if numberTrue == 0 and numberFalse > 0:
		 	node = Node('value',str(self._nodeCount))
			self._nodeCount += 1
		 	node.setValue(0.0)
		 	self._nodes[node.getId()] = node
			##print 'false node'
			#raw_input()
			return node

		
		#*If no features, create percentage node true from remaining data !WILL BE CALCULATION ERRORS!
		if len(featuresList) == 0:
			node = Node('value',str(self._nodeCount))
			self._nodeCount += 1
			node.setValue(float(numberTrue)/(numberTrue+numberFalse))
			self._nodes[node.getId()] = node
			##print 'percentage node'
			#raw_input()
			return node

#---------------------------------------------------------------------------------------------------------------------
		#*The tree grows
		#** Create this node, and add to node dictionary
		node = Node('Branching',str(self._nodeCount))
		node.setValue('Branching')
		self._nodeCount += 1
		self._nodes[node.getId()] = node
		
		#**Calculate entropy for current data
		totalNumber  = numberTrue+numberFalse                # total number of data
		totalEntropy = calc_entropy(totalNumber,numberTrue)  # entropy for total data
		
		#**Find feature with highest gain
		currentFeatureIndex = 0                        # list iterator
		maxFeatureGain = 0                             # hold highest feature gain found
		numberOfFeatures = len(featureData)						 # number of features, 131 this case
		while(currentFeatureIndex < len(featureData)): # loop through features
			rowOfFeatureValues = featureData[currentFeatureIndex] # place current row of 
																														#  features into variable
			#***Skip past entries that are empty until non empty cell is found
			checkTypeIndex = 0
			while rowOfFeatureValues[checkTypeIndex] == '' and checkTypeIndex < len(rowOfFeatureValues)-1:
				checkTypeIndex+=1

			#***Determine data type of feature (number or category) and calculate gain
			if is_number(rowOfFeatureValues[checkTypeIndex]):        # if 'number'
				sortedValues = list(rowOfFeatureValues)                # sort values and remove missing data
				sortedValues.sort()															       # -
				sortedValues = filter(lambda a: a != '', sortedValues) # -

				#***find highest gain from splitting list at each value in sortedValues
				sortedValueIndex = 0
				lengthSortedValues = len(sortedValues)
				gainList = []
				sortingValueList = []
				floatRowTotalsList = []
				floatRowTrueTotals = []
				while(sortedValueIndex < len(sortedValues)):
					overNumberTrue = overNumberFalse = 0
					underNumberTrue = underNumberFalse = 0
					emptyQuotesNumberTrue = emptyQuotesNumberFalse = 0
					# split data into three groups: 
					#  1. > current sorting value
					#  2. <= current sorting value 
					#  3. empty quotes (missing data)
					# get number and true ('1') and false ('0') for each group 

					currentSortingValue = sortedValues[sortedValueIndex]
					dataItemIndex = 0
					lengthDataItems = len(membershipData)
					while(dataItemIndex < len(membershipData)):
						dataItem = rowOfFeatureValues[dataItemIndex] 
						if dataItem != '':
							if dataItem > currentSortingValue:
								if membershipData[dataItemIndex] == '1':
									overNumberTrue += 1
								else:
									overNumberFalse += 1
							else: # lower than or equal to currentSortingValue
								if membershipData[dataItemIndex] == '1':
									underNumberTrue += 1
								else:
									underNumberFalse += 1
						else:
							if membershipData[dataItemIndex] == '1':
								emptyQuotesNumberTrue += 1
							else:
								emptyQuotesNumberFalse += 1
						dataItemIndex += 1

					#calculate gain at current splitting value
					rowTotals = [overNumberTrue+overNumberFalse, underNumberTrue+underNumberFalse, emptyQuotesNumberTrue+emptyQuotesNumberFalse]
					rowTrueTotals = [overNumberTrue, underNumberTrue, emptyQuotesNumberTrue]
					
					gain = calc_gain(totalEntropy, totalNumber, rowTotals, rowTrueTotals)

					gainList.append(gain)
					sortingValueList.append(currentSortingValue)
					floatRowTotalsList.append(rowTotals)
					floatRowTrueTotals.append(rowTrueTotals)
					sortedValueIndex+=1
				
				# If this is current highest gain replace previous feature data with this data
				if len(gainList) > 0 and max(gainList) > maxFeatureGain:
					maxFeatureGain = max(gainList)
					maxFeatureGainIndex = gainList.index(maxFeatureGain)

					maxGainSortingValue       = float(sortingValueList[maxFeatureGainIndex])
					maxGainRowTotals     = floatRowTotalsList[maxFeatureGainIndex]
					maxGainRowTrueTotals =  floatRowTrueTotals[maxFeatureGainIndex]
					maxGainFeatureIndex =  currentFeatureIndex 
				
				#plt.plot(gainList)
				#plt.suptitle(featuresList[currentFeatureIndex])
				#plt.show()

			else: #----------------------------------------------# category
				rowOfFeatureValues = featureData[currentFeatureIndex]

				valueDict = {} # total totalTrue totalFalse
				currentFeatureIndexValueLine = 0
				lengthRowOfFeatureValues = len(membershipData)
				while( currentFeatureIndexValueLine < len(membershipData)):
					line = rowOfFeatureValues[currentFeatureIndexValueLine]
					if valueDict.has_key(line) == False:
						valueDict[line] = [1,0,0]
						if membershipData[currentFeatureIndexValueLine] == '1':
							valueDict[line][1] = 1
							valueDict[line][2] = 0
						else:
							valueDict[line][1] = 0
							valueDict[line][2] = 1
					else:
						valueDict[line][0] += 1
						if membershipData[currentFeatureIndexValueLine] == '1':
							valueDict[line][1] += 1
						else:
							valueDict[line][2] += 1
					
					currentFeatureIndexValueLine+=1
				#calculate Gain for valueDict
				'''
				#
				#
				# Probem in here--------------------------------------------------------------
				#
				# remove the data for valueDict from training data
				#
				#
				if(len(valueDict)) > 5 :
					#print valueDict
					gain = .00001

					#print '!!!!!!!!!!!'
					#print currentFeatureIndex
					#print len(featuresList)
					del featuresList[currentFeatureIndex]
					#print len(featuresList)
					
					#print np.shape(featureData)
					#print featureData[0][21]
					#print featureData[0][22]

					featureData = np.delete(featureData,currentFeatureIndex,1)
					#print np.shape(featureData)
					#print featureData[0][21]
					raw_input('HERE')
					# remove data from featuresList,membershipData,featureData
				else:
				'''
				#
				#
				# -----------------------------------------------------------------------------
				#
				#

				#---------------------------------------------
				#if(len(valueDict)) > 10:
				#	#print currentFeatureIndex

				rowTotals = []
				rowTrueTotals = []
					
				for entry in valueDict:
					tempEntry = valueDict[entry]
					rowTotals.append(tempEntry[0])
					rowTrueTotals.append(tempEntry[1])
				gain = calc_gain(totalEntropy, totalNumber, rowTotals, rowTrueTotals)
				
				if gain > maxFeatureGain:
					maxFeatureGain = gain

					maxGainSortingValue       = 'char'
					maxGainRowTotals     = rowTotals
					maxGainRowTrueTotals =  rowTrueTotals
					maxGainFeatureIndex =  currentFeatureIndex 
					maxGainValueDict = valueDict

			currentFeatureIndex+=1
		

		#**For the feature with highest gain
		#  sort pieces of data into groups from feature _note:treat quotes as a group in both cases
		#  if categorical sort into categories from feature
	
		highestGainFeatureData = featuresList[maxGainFeatureIndex] # label of feature with highest gain

		if maxGainSortingValue == 'char':
			##print maxGainValueDict  
			##print '************'
			##print maxGainValueDict

			highestOdds = 0			
			groupWithHighestOdds = ''
			for group in maxGainValueDict:
				# calculate group that has highest odds of data being examined belonging to it
				#  this is testing data that has a categorical value that was not seen during tree construction 
				groupOdds = float( maxGainValueDict[group][1] ) / maxGainValueDict[group][0]
				
				if groupOdds > highestOdds:
					highestOdds = groupOdds
					groupWithHighestOdds = group

				groupFeaturesList = []
				groupMembershipData = []
				groupFeatureData = []

				# get list of feature labels and place into groupFeaturesList, omit feature label of feature with highest gain
				groupFeaturesList = list(featuresList)
				del groupFeaturesList[maxGainFeatureIndex]


				# get group membership data
				lengthMembershipData = len(membershipData)
				dataIndex = 0
				while dataIndex < lengthMembershipData:
					if group == featureData[maxGainFeatureIndex][dataIndex]:
						groupMembershipData.append(membershipData[dataIndex])
					dataIndex+=1

				# get group feature data
				lengthFeatureData = np.shape(featureData)[1]
				dataIndex = 0
				while dataIndex < lengthFeatureData:					
					if group == featureData[maxGainFeatureIndex][dataIndex]:
						#group FeatureData
						individualData = []
						tempIndex = 0
						while tempIndex <  int(np.shape(featureData)[0]):
							individualData.append(featureData[tempIndex][dataIndex])
							tempIndex+=1						
						groupFeatureData.append(individualData)
 
					dataIndex+=1
				groupFeatureData = np.transpose(groupFeatureData)
				# remove features used to calculate this gain from feature data set
				groupFeatureData = np.delete(groupFeatureData,maxGainFeatureIndex,0)  
#------>#!!!Pass the groupFeatureData groupFeaturesList groupMembershipData to recursive function here
				tempNode = self.CreateNode(groupFeaturesList,groupMembershipData,groupFeatureData,overAllOddsOfMembership)
				tempNode.setAddress(group)
				node.addBranch(tempNode.getId())
			node.setFeatureIndex(maxGainFeatureIndex)
			node.setBranchType('category')
			node.setGroupWithHighestOdds(groupWithHighestOdds)
		#if numerical sort by splitting value that gives highest gain
		else:
			underGroupFeatureData = []
			underGroupMembershipData = []
			overGroupFeatureData = []
			overGroupMembershipData = []
			emptyGroupFeatureData = []
			emptyGroupMembershipData = []
			groupFeaturesList = []

			# get list of feature labels and place into groupFeaturesList, omit feature label of feature with highest gain
			groupFeaturesList = list(featuresList)
			del groupFeaturesList[maxGainFeatureIndex]

			#groupFeatureData,groupMembershipData
			lengthFeatureData = len(featureData[maxGainFeatureIndex])
			dataIndex = 0
			while dataIndex < lengthFeatureData:
				if featureData[maxGainFeatureIndex][dataIndex] == '':
					individualData = []
					tempIndex = 0
					while tempIndex < len(featureData):
						if tempIndex !=  (int(highestGainFeatureData[1:])-1):
							individualData.append(featureData[tempIndex][dataIndex])
						tempIndex+=1 
					emptyGroupFeatureData.append(individualData)
					emptyGroupMembershipData.append(membershipData[dataIndex])
				
				else:
					currentComparingValue = float(featureData[maxGainFeatureIndex][dataIndex])
					if currentComparingValue <= maxGainSortingValue:
						#group FeatureData
						individualData = []
						tempIndex = 0
						while tempIndex < len(featureData):
							if tempIndex !=  (int(highestGainFeatureData[1:])-1):
								individualData.append(featureData[tempIndex][dataIndex])
							tempIndex+=1 
						underGroupFeatureData.append(individualData)
						underGroupMembershipData.append(membershipData[dataIndex])

					elif currentComparingValue > maxGainSortingValue:
						#group FeatureData
						individualData = []
						tempIndex = 0
						while tempIndex < len(featureData):
							if tempIndex !=  (int(highestGainFeatureData[1:])-1):
								individualData.append(featureData[tempIndex][dataIndex])
							tempIndex+=1 
						overGroupFeatureData.append(individualData)
						overGroupMembershipData.append(membershipData[dataIndex])
				dataIndex+=1

			node.setBranchType('float')
			node.setFeatureIndex(maxGainFeatureIndex)
			node.setSplitValue(maxGainSortingValue)

 			tempNode = self.CreateNode(groupFeaturesList,underGroupMembershipData,np.transpose(underGroupFeatureData),overAllOddsOfMembership)
 			tempNode.setAddress('under')
			##print self._nodeCount
			node.addBranch(tempNode.getId())

 			tempNode = self.CreateNode(groupFeaturesList,overGroupMembershipData,np.transpose(overGroupFeatureData),overAllOddsOfMembership)
 			tempNode.setAddress('over') 
			##print self._nodeCount
			node.addBranch(tempNode.getId())

 			tempNode = self.CreateNode(groupFeaturesList,emptyGroupMembershipData,np.transpose(emptyGroupFeatureData),overAllOddsOfMembership)
 			tempNode.setAddress('empty')
 			##print self._nodeCount
			node.addBranch(tempNode.getId())

		return node

	def Test(self,testData,testIds):
		'''
		#print self._nodes['0'].getType()		
		#print self._nodes['0'].getBranchType()
		#print self._nodes['0'].getFeatureIndex()
		'''
		resultsStr = ''
		countForIdRetrieval = 0
		for line in testData:
			currentNodeId = '0'
			currentNode = self._nodes[currentNodeId] # root node
			while currentNode.getValue() == 'Branching':
				branchType = currentNode.getBranchType()
				#print branchType
				featureIndex = currentNode.getFeatureIndex()
				
				if branchType == 'category':
					#print currentNode.getPaths()
					dataNotSeenInGroups = True
					for path in currentNode.getPaths():
						#print path
						#print self._nodes[path].getAddress()
						#print featureIndex
						#print line[featureIndex]
						#raw_input('-------')
						
						if currentNode.getGroupWithHighestOdds() == self._nodes[path].getAddress():
							highestOddsPath = path

						if line[featureIndex] == self._nodes[path].getAddress():
							currentNode = self._nodes[path]
							line = np.delete(line,featureIndex,0)
							dataNotSeenInGroups = False
							#print path
							#print '****************'
							break
					if dataNotSeenInGroups == True:
						#print currentNode.getGroupWithHighestOdds() 
						currentNode = self._nodes[highestOddsPath]
						line = np.delete(line,featureIndex,0)
				elif branchType == 'float':
					splitValue =  currentNode.getSplitValue()
					#print splitValue
					#print featureIndex
					#print line[featureIndex]

					if line[featureIndex] == '': # empty quotes
						#print 'empty'
						for path in currentNode.getPaths():
							if self._nodes[path].getAddress() == 'empty':
								currentNode = self._nodes[path]
					else: # not empty quotes
						if float(line[featureIndex]) <= splitValue: # under
							#print 'under'
							for path in currentNode.getPaths():
								if self._nodes[path].getAddress() == 'under':
									currentNode = self._nodes[path]
									break
						else :  # over
							#print 'over'
							for path in currentNode.getPaths():
								if self._nodes[path].getAddress() == 'over':
									currentNode = self._nodes[path]
									break
						#print featureIndex
						line = np.delete(line,featureIndex,0)
					#raw_input('-------------')	
				else: # error
					#print 'Error: class:Tree -> function:Test -> branchType unrecognized'
					quit()
			#print 'predicted value'
			#print currentNode.getValue()
			#print type(currentNode.getValue())
			resultsStr += str(testIds[countForIdRetrieval])+','+str(currentNode.getValue())+'\n'
			countForIdRetrieval += 1
			#raw_input()

		return resultsStr	
		
	def PrintTree(self,nodeId,depth):
		if self._nodes.has_key(nodeId):
			if self._nodes[nodeId].getType() == 'Branching':
				paths =  self._nodes[nodeId].getPaths()
				#print paths
				#print '-'*10
				for path in paths:
					#print str(depth)+' '+path+'---->path'
					self.PrintTree(path,depth+1)
				return
			else:
				#print str(depth)+' '+str(self._nodes[nodeId].getValue())
				return
		else:
			return
		

		
#******************************************************************



if __name__ == '__main__':
	f = open('train.csv','r')
	fileData = f.read().split('\n')[:-1]                          # read file data into a list, remove last line
	fileData = fileData[: -( len(fileData)-len(fileData)/10 ) ] # grab subsection of full data from file
	##print len(fileData)
	f.close()
	
	# Break fileData list members into lists themselves
	# place in list named data
	data = []
	for line in fileData:
		line = line.split(',')
		del line[23]
		del line[52]
		del line[55]
		del line[77]
		del line[109]
		del line[109]
		del line[120]
		#quit()
		data.append(line)
		

#---------------IN PROGRESS
	#data= data[:-2]
	#splitData = np.split(np.array(data),112/14.0,0)
#--------------------------

	#featuresList   -> Label (feature name) for each column of data 
	#membershipData -> Classification (target answer) for individual data
	#dataT          -> Transposed member data, each line is list of members data for a single feature
	featuresList = data[0][2:] # omit 'ID' and 'Target'

	data = data[1:]            # omit feature list from data

	dataT = np.transpose(data) 
	membershipData =  dataT[1] 
	dataT = dataT[2:] 				 #omit id and target from member data

	# Create Tree

	print 'beginning training'
	tree = Tree(featuresList,membershipData,dataT)

	f = open('test.csv')
	testFileData = f.read().split('\n')[:-1]
	f.close()
	testData = []
	for line in testFileData:
		line = line.split(',')
		del line[22]
		del line[51]
		del line[54]
		del line[76]
		del line[108]
		del line[108]
		del line[119]
		testData.append(line)
 
	#-------------------------------------------
	testFeaturesList = testData[0][:1] # omit ID

	testData = testData[1:]            # omit feature list ftom test data
	testDataT = np.transpose(testData)
	testIds = testDataT[0]
	testDataTT = np.transpose(testDataT[1:])

	print 'beginning test...'
	results = tree.Test(testDataTT,testIds)
	print 'writing results to file.'
	f = open('decisionTreeResults.csv','w')
	fileStr = 'ID,PredictedProb\n'+results
	f.write(fileStr)
	f.close()

