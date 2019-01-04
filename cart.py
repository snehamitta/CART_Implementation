import matplotlib.pyplot as plt
import numpy as np
import pandas
import itertools


data = pandas.read_csv('/Users/snehamitta/Desktop/ML/Assignment3/CustomerSurveyData.csv',
                       delimiter = ',')

def GiniForRoot(data, target):

    
    nTotal = len(data)
    crossTable = pandas.crosstab(index = data[target], columns = ['Count'], margins=True, dropna = False)
    crossTable['Percent'] = (crossTable['Count'] / nTotal)
    crossTable = crossTable.drop(columns = ['All'])
    
    nRows = crossTable.shape[0]
        
    perportion = 0
    for nRow in range(nRows-1):
        perportion = perportion + crossTable.iloc[nRow, -1]**2
    gini = 1-perportion
    print("The Gini for root is", gini)

def GiniSplit (data, target, predictor, split):

    print(split)
    crossTable = pandas.crosstab(index = data[predictor].fillna('Missing'), columns = data[target], margins=True, dropna = False)
    tempTable = pandas.DataFrame(columns=crossTable.columns)
    for i in split:
        tempTable.loc[i] = crossTable.loc[i]
        crossTable = crossTable.drop(i);
    crossTable = crossTable.drop("All")
    table = pandas.DataFrame(columns=crossTable.columns)
    table.loc["RIGHT_SPLIT"] = crossTable.apply(lambda x: x.sum(), axis=0)
    table.loc["LEFT_SPLIT"] = tempTable.apply(lambda x: x.sum(), axis=0)
    table.loc["ALL"] = table.apply(lambda x: x.sum(), axis=0)
    
    nColumns = table.shape[1]
    
    gini1 = 0
    for nColumn in range(nColumns-1):
        gini1 = gini1 + (table.loc["RIGHT_SPLIT"][nColumn]/table.loc["RIGHT_SPLIT"][-1])**2
    gini1 = 1-gini1
    
    gini2 = 0
    for nColumn in range(nColumns-1):
        gini2 = gini2 + (table.loc["LEFT_SPLIT"][nColumn]/table.loc["LEFT_SPLIT"][-1])**2
    gini2 = 1-gini2
    
    splitGini = gini1*table.loc["RIGHT_SPLIT"][-1]/table.iloc[-1,-1] + gini2*table.loc["LEFT_SPLIT"][-1]/table.iloc[-1,-1]
                
    print(table)
    print("Right Split", gini1)
    print("Left Split", gini2)
    print("Split Gini", splitGini)


#Q1(a)       
GiniForRoot(data, "CreditCard")

#Q1(b) 
bsplits = 2**(3-1) - 1
print ('The possible number of binary splits for CarOwnership Predictor are', bsplits)

#Q1(c)
category = ["None","Lease","Own"]
for i in itertools.combinations(category,1):
    GiniSplit(data=data, target="CreditCard", predictor="CarOwnership", split =list(i))

#Q1(e)
bsplits = 2**(7-1) - 1
print ('The possible number of binary splits for JobCategory Predictor are', bsplits)

#Q1(f)
category = ["Agriculture", "Crafts", "Labor","Professional","Sales","Service","Missing"]
for i in itertools.combinations(category,1):
    GiniSplit(data=data, target="CreditCard", predictor="JobCategory", split =list(i))
for i in itertools.combinations(category,2):
    GiniSplit(data=data, target="CreditCard", predictor="JobCategory", split =list(i))
for i in itertools.combinations(category,3):
    GiniSplit(data=data, target="CreditCard", predictor="JobCategory", split =list(i))






        
        

