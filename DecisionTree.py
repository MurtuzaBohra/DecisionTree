arg_A = '';
arg_B = '';

import csv
import numpy as np
import sys
files=sys.argv
files.pop(0)


datareader = csv.reader(open(files[0], "rb"), delimiter=",")
temp = datareader.next()

datareader1 = csv.reader(open(files[1], "rb"), delimiter=",")
temp_test = datareader1.next()


y_name='left'
max_split=15
diffrence=0.0001 #entropy difference to increase split for continuous attribute
percent_val=0

cont_col=[]#list of headers of features having continues numeric values.


header = []
test_header=[]
DATA = []
TEST = []
attributes_struct=[]

for name in temp:
	header.append(name)

for name in temp_test:
	test_header.append(name)

y_ind=header.index(y_name)  

			
for row in datareader:
	DATA.append(row)

for row in datareader1:
	TEST.append(row)

TEST=np.array(TEST)
DATA=np.array(DATA)

val_len=int(DATA.shape[0]*(float(percent_val)/100))
traindata_len=int(DATA.shape[0]-val_len) 
traindata=DATA[0:traindata_len,:]
val_data=DATA[traindata_len:DATA.shape[0],:]


#**********************Slicing of continuous attributes***********************************
# If the data has continues numeric features then 'slice' is calculated. 
#'slice' is used to descritize the continues feature into small ranges.

slice = np.zeros((3,len(cont_col)),dtype=float)

def Entropy(a,b):
	if a==0 and b==0:
		return 0.0
	else:
		return ((float(a)/(a+b))**2) + ((float(b)/(a+b))**2)


for col in cont_col:
	ind=header.index(col)
	minimum=min(traindata[:,ind].astype(float))
	maximum=max(traindata[:,ind].astype(float))
	slice[1,cont_col.index(col)]=float(minimum)
	slice[2,cont_col.index(col)]=float(maximum)

	  
	avg_ig=0.0

	for n in range(2,max_split+1):
		count=np.zeros((2,n),dtype=int)
		w=float(maximum-minimum)/n
		for j in range(0,traindata.shape[0]):
			for i in range(1,n+1):
				if float(traindata[j,ind]) >= ((i-1)*w)+minimum and float(traindata[j,ind]) <= ((i)*w)+minimum:
					if traindata[j,y_ind]=='0':
						count[0,i-1]+=1
					else:
						count[1,i-1]+=1
		entropy_sum=0
		
		for i in range(0,n):
			entropy_sum = entropy_sum + Entropy(count[0,i],count[1,i])*(float(count[0,i]+count[1,i])/traindata.shape[0])

		if (entropy_sum-avg_ig)>diffrence:
			slice[0,cont_col.index(col)]=n
			avg_ig=entropy_sum


#********************************************************

def create_attributes_struc():
	for name in header:
		if name in cont_col:
			temp=[]
			ind=cont_col.index(name)
			for i in range(0,int(slice[0,ind])):
				temp.extend([slice[1,ind]+((i+1) * ((slice[2,ind] - slice[1,ind])/slice[0,ind]))])
			attributes_struct.append(temp)
		else:
			ind=header.index(name)
			attributes_struct.append(list(set(traindata[:,ind])))




def Split_entropy(Data, attr):
	ind=header.index(attr)
	split=len(attributes_struct[ind])
	count=np.zeros((2,split),dtype='int')
	newData=[[] for i in range(split)]
	
	if attr in cont_col:
		for j in range(0,len(Data)):
			for i in range(0,split):
				if round(float(Data[j][ind]), 2) <= round(float(attributes_struct[ind][i]),2):
					newData[i].append(Data[j])
					if int(Data[j][y_ind])==0:
						count[0,i]+=1
					else:
						count[1,i]+=1
					break;
	else:
		for j in range(0,len(Data)):
			for i in range(0,split):
				if Data[j][ind] == attributes_struct[ind][i]:
					flag=False
					newData[i].append(Data[j])
					if int(Data[j][y_ind])==0:
						count[0,i]+=1
					else:
						count[1,i]+=1
					break;

	entropy_sum=0

	for i in range(0,split):
		entropy_sum=entropy_sum + (float(count[0,i]+count[1,i])/len(Data)) * Entropy(count[0,i],count[1,i])
	return entropy_sum,count,newData




def max_split_entropy(Data,used_attr):
	max_entropy=Entropy([row[y_ind]for row in Data].count('0'),[row[y_ind]for row in Data].count('1'))
	count=[]
	newData=0
	attr=None
	for name in (set(header)- set(used_attr)):
		temp_entropy,temp_count,temp_newData = Split_entropy(Data,name)
		if max_entropy<temp_entropy:
			max_entropy=temp_entropy
			count=temp_count
			newData=temp_newData
			attr=name
	return max_entropy,count,newData,attr


class node:
	child=[]
	name=None
	count_class0=0
	count_class1=0
	def __init__(self, a,b):
		self.count_class0=a
		self.count_class1=b
		self.child = []
		self.name=None


def decision_tree(Node,Data,used_attr, cn):
	max_entropy,count,newData,attr=max_split_entropy(Data, used_attr)
	if attr !=None:		
		Node.name=attr
		used_attr.append(attr)

		for i in range(len(attributes_struct[header.index(attr)])):
			if len(newData[i])!=0:
				temp_node=node(count[0,i],count[1,i])
				Node.child.append(temp_node)
				if max_entropy!=1:
					temp_used_attr=[]
					temp_used_attr = [z for z in used_attr]
					decision_tree(Node.child[i],newData[i], temp_used_attr, cn +str(i))
			else:
				Node.child.append(None)
	else:
		Node.child=None


#********************testing******************


def decide(Node):
	result=0
	if Node.count_class0>=Node.count_class1:
		result=0
	else:
		result=1
	return result


def test(Node,row):
	result=0;
	if Node.name==None:
		result=decide(Node)

	else:
		flag= True
		ind=header.index(Node.name)
		ind_test=test_header.index(Node.name)
		if Node.name in cont_col:
			for i in range(len(attributes_struct[ind])):
				if float(row[ind_test])<=float(attributes_struct[ind][i]):
					flag=False
					if Node.child[i]==None:
						result=decide(Node)
					else:
						result=test(Node.child[i],row)
					break;
			if flag:
				result=decide(Node)

		else:
			for i in range(len(attributes_struct[ind])):
				if row[ind_test] == attributes_struct[ind][i]:
					flag=False
					if Node.child[i]==None:
						result=decide(Node)
					else:
						result=test(Node.child[i],row)
					break;
				if flag:
					result=decide(Node)
	return result




#---------------------------------------------------------------------
#---------------------Main function-----------------------------------

attributes_struct = create_attributes_struc()

data=traindata.tolist()
root=node([row[y_ind]for row in data].count('0'),[row[y_ind]for row in data].count('1'))
used_attribute=list([y_name])
decision_tree(root,data,used_attribute, '0')

TEST=TEST.tolist()

for row in TEST:
	x=test(root,row)
	print x;

'''
print 'accuracy= ',1-(float(mis)/len(val_data))
val_data=val_data.tolist()
mis=0
for row in val_data:
	x=test(root,row)
	if int(row[y_ind])!=x:
		mis+=1
'''
#---------------------------------------------------------------------
#---------------------------------------------------------------------
