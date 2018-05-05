
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


def clean_data(data_frame):
    cols = data_frame.columns.values
    # For nominal attributes, if variable contains either 2 or 3 unique values, replace NaN with mode of the attribute
    for i in cols:
        if (data_frame[i].nunique() < 4):
            l = [0,1, np.nan]
            if not all(train[i].isin(l)):
                data_frame[i] = data_frame[i].fillna(data_frame[i].mode())
        # For attributes with "object" datatype, factorize the data column
        if data_frame[i].dtypes == object:
            data_frame[i] = pd.factorize(data_frame[i])[0]
        # Variables with continuous values in range of 0 to 1, are replaced by mean
        if (data_frame[i].nunique() > 4) and (data_frame[i].count() > len(data_frame.index)/1.1) and (data_frame[i].dtypes != object):
            # Variables with continuous values in range of 0 to 1, Nan values are replaced by mean
            if (data_frame[i].max() <= 1):
                data_frame[i] = data_frame[i].fillna(data_frame[i].mean())
            else:
                # Continuous variables with range more than 1 is replaced by mode
                data_frame[i] = data_frame[i].fillna(data_frame[i].mode())
    data_frame.fillna(-1, inplace=True)
    return data_frame


# In[3]:


train = pd.read_csv("training.csv")
df_all = train
df = clean_data(df_all)
#df.fillna(-1, inplace=True)
cols = list(df.columns.values)
cols.pop(cols.index('Response'))
df = df[cols+['Response']]


# In[4]:


df.to_csv("cleaned_training.csv")


# In[5]:


test_data = pd.read_csv("testing.csv")
main_solution = pd.DataFrame(test_data['Id'])
test_data = test_data.drop(['Id'], axis = 1)


# In[6]:


test_data1 = clean_data(test_data)
#test_data1.fillna(-1, inplace=True)


# In[7]:


test_data1.to_csv('cleaned_testing.csv')


# In[8]:


#The Attribute class which includes the column number and the value for the column number
#This class contains the value of the attribute which is used to divide the dataset at each node of the tree
class attribute:
    def __init__(self,value,index):
        self.value = value
        self.index = index   
        


# In[9]:


#This function compares an attribute x with another attribute to see 
#what side of the tree the given attribute x should belong to
def compare_attribute(x,attribute):   
    compare_to = x[attribute.index]
    if ((type(compare_to)== type(7))or(type(compare_to)==type(7.7))):
        return (compare_to >= attribute.value)
    else:
        return (compare_to == attribute.value)


# In[10]:


#This function divides the data based on the attribute
def divide_data(data,attribute):
    left_data = []
    right_data = []
    for x in data:
        if compare_attribute(x,attribute): 
            right_data.append(x)
        else:
            left_data.append(x)
    return left_data,right_data  
            


# In[11]:


#Calculates gini index of given data
def gini(data):
    temp = []
    count_temp = []
    for x in data:
        if (x[len(x)-1] not in temp):
            temp.append(x[len(x)-1])
            count_temp.append(0)
        count_temp[temp.index(x[len(x)-1])] += 1
    out = 1;
    for i in range(0,len(temp)):
        curr_probability = count_temp[i]/len(data)
        out = out - curr_probability**2
    return out
            


# In[12]:


#Calculates the reduction in gini index after the division has been performed
def gini_reduction(initial_gini,left_data,right_data):
    probability_left = len(left_data)/(len(left_data)+len(right_data))
    probability_right = 1-probability_left
    out = initial_gini-probability_left*gini(left_data)-probability_right*gini(right_data)
    return out


# In[13]:


#Finds the best division attribute for the given data based on gini index
def division_attribute(data):
    output_attribute = None
    max_gini_reduction = 0;
    initial_gini = gini(data)
    for i in range(0,len(data[0])-1):
        column_unique_values = []
        for x in data:
            if (x[i] not in column_unique_values):
                column_unique_values.append(x[i])
        for y in column_unique_values:
            curr_attribute = attribute(y,i)
            left_data,right_data = divide_data(data,curr_attribute)
            curr_gini_reduction = gini_reduction(initial_gini,left_data,right_data)
            if ((curr_gini_reduction >= max_gini_reduction)and(curr_gini_reduction != 0)):
                max_gini_reduction = curr_gini_reduction
                output_attribute = curr_attribute
    return output_attribute
    


# In[14]:


#Counts instances of all classes in the given dataset
def count_instances(data):
    out = {}
    for x in data:
        temp = x[len(x)-1]
        if temp not in out:
            out[temp] = 1
        else:
            out[temp] += 1
    return out


# In[15]:


#The Node Datastructure implementatioin for the Decision Tree
class Node:
    def __init__(self, left_branch, right_branch, attribute, data):
        self.left_branch = left_branch
        self.right_branch = right_branch
        self.attribute = attribute
        if (attribute == None): self.content = count_instances(data) 


# In[16]:


#Construction of a decision tree from given data
def make_decision_tree(data):
    if (len(data) <= 30): return Node(None,None,None,data)
    attribute = division_attribute(data)
    if (attribute == None): return Node(None,None,None,data)
    left_data, right_data = divide_data(data, attribute)
    left_branch = make_decision_tree(left_data)
    right_branch = make_decision_tree(right_data)
    return Node(left_branch, right_branch, attribute,data)
    


# In[17]:


#Predicts the class of sample from the tree with root node
def predict(sample, node):
    if (node.attribute == None):
        return node.content
    if compare_attribute(sample,node.attribute):
        return predict(sample, node.right_branch)
    else:
        return predict(sample, node.left_branch)


# In[18]:


#Makes a dataset based on the weights
import random
def make_data_set(training_data,weights):
    curr_set = []
    for i in range(0,len(training_data)):
        random_number  = random.uniform(0,1)
        sum = 1/len(training_data)
        for j in range(0,len(weights)):
            if (random_number < sum):
                curr_set.append(training_data[j])
                break
            sum +=1/len(training_data)    
    return curr_set
    


# In[19]:


#Merges the prediction results of multiple trees
def merge(dicts):
    out = {}
    for data in dicts:
        for x in data.keys():
            if x not in out:
                out[x] = 0
            out[x] += data[x] 
    return out


# In[20]:


#Returns a class based on multiple prediction dictionaries
def predict_result(dict_predictions):    
    curr_dict = dict_predictions
    out = 0 
    num = 0
    for x in curr_dict.keys():
        if (curr_dict[x]>num):
            out = int(x)
            num = curr_dict[x]
    return out


# In[21]:


#Assigns votes to the classifier results
def process(dict,vote):
    out  = {}
    for x in dict.keys():
        out[x]= dict[x]*vote
    return out


# In[22]:


# def get_label_weights(training_data):
#     a = count_instances(training_data)
#     for x in a.keys():
#         a[x] = len(training_data)/a[x]
#     return a


# In[23]:


# def multiply_dicts(a,b):
#     out = {}
#     for x in a.keys():
#         out[x] = a[x]*b[x]
#     return out    


# In[24]:


#Calculates gini index of the predictions results dictionary
def gini_dict(dict_):
    temp = dict_
    sum_temp = 0
    for x in temp.keys():
        sum_temp+=temp[x]
    out  = 1
    for x in temp.keys():
        temp[x] = temp[x]/sum_temp
        out = out - temp[x]**2
    return out


# In[25]:


#Returns the average of all classes in the given dataset
def predict_average(dict_predictions):    
    curr_dict = dict_predictions
    out = 0 
    num = 0            
    for x in curr_dict.keys():
            out = out+x*curr_dict[x]
            num = num+curr_dict[x]
    out = int(round((out/num-1)*8/7,1))
    if (out>8): out = 8
    if (out<1): out = 1
    return out


# In[26]:


# def check_and_predict_average(dict_predictions):    
#     curr_dict = dict_predictions
#     out = 0 
#     num = 0
#     for x in curr_dict.keys():
#         c = 0
#         for y in curr_dict.keys():
#             if((x!=y)and(curr_dict[x]<2*curr_dict[y])): c = 1
#         if (c==0): return int(x)        
                
#     for x in curr_dict.keys():
#             out = out+x*curr_dict[x]
#             num = num+curr_dict[x]
#     #print("Out = %s and Num = %s" %(out,num))        
#     out = int(round((out/num-1)*8/7,1))
#     if (out>8): out = 8
#     if (out<1): out = 1
#     return out


# In[27]:


##Reading from the file
fo = open("cleaned_training.csv", "r")

a = fo.readline()
a = a.strip().split(',')
a = a[2:]
Column_names = a
training_data = []
for i in range(0,20000):
    temp = fo.readline()
    temp = temp.strip().split(',')
    temp = temp[2:]
    temp = [round(float(a),1) for a in temp]
    training_data.append(temp)    


# In[28]:


##Assigning Weights for AdaBoost
weights = [1/len(training_data)]*len(training_data)


# In[29]:


#making 10 trees using AdaBoosting
import math
trees = [] #The trees arrays
votes = [] #The votes of each tree array

num_trees = 10

for j in range(0,num_trees):
    print("Starting Classfier number: "+str(j))
    curr_set = make_data_set(training_data,weights) #Get the dataset based on weights
    curr_tree = make_decision_tree(curr_set) #Make decision tree from the dataset
    trees.append(curr_tree)
    print("Tree Constrcuted")
    error = 0
    correctness_array = []
    for i in range(0,len(training_data)): #Checking predictions for the training set to calculate error
        
        d = []
        for k in range(0,j+1):
            temp = predict(training_data[i],trees[k])
            d.append(temp)
        oo = merge(d)
        if (gini_dict(oo)<0.4):
             x= predict_result(oo)
        else:
            x = predict_average(oo) #x is the predicted class for the current tuple
        #temp = predict(row,curr_tree) 
        actual = int(training_data[i][-1])
        if (actual!=x):
            error = error+ weights[i]*((abs(actual-x)/7)**2)
            correctness_array.append(-(abs(actual-x)/7)**2)
        else:
            correctness_array.append(1)
    print("Error= "+str(error))        
    curr_vote = 0.5*math.log((1-error)/error) ##Calculating vote for current classifier
    votes.append(curr_vote)
    print("Vote= "+str(curr_vote))
    for i in range(0,len(weights)):
        weights[i] = weights[i]*math.exp(-curr_vote*correctness_array[i]) ##Updating weights
    sum_weights = sum(weights)
    print("Sum of weights = " + str(sum_weights))
    weights = [a/sum_weights for a in weights] ##Normalizing weights       


# In[30]:


# #checking the accuracy for training set(first 100 tuples)
# #label_weights = get_label_weights(training_data)
# for row in training_data[:100]:
#     d = []
#     for i in range(0,num_trees):
#         temp = predict(row,trees[i])
#         #temp = process(temp,votes[i])
#         d.append(temp)
#     oo = (merge(d))
#     #oo = multiply_dicts(oo,label_weights)
#     ll = predict_average(oo)
#     print(gini_dict(oo))
#     if (gini_dict(oo)<0.4):
#         ll = predict_result(oo)
#     else:
#         ll = predict_average(oo)
#     print ("Actual: %s. Predicted: %s and result: %s" %
#            (row[-1], oo,ll))


# In[31]:


#Reading the test dataset
f2 = open("cleaned_testing.csv","r")
testing_data = []
header = f2.readline()
for i in range(0,10000):
    temp = f2.readline()
    temp = temp.strip().split(',')
    temp = temp[1:]
    temp = [round(float(a),1) for a in temp]
    testing_data.append(temp) 
len(testing_data)


# In[32]:


#label_weights = get_label_weights(training_data)
dict_predictions = []
for row in testing_data:
    d = []
    
    for i in range(0,num_trees):
        temp = predict(row,trees[i])
        temp = process(temp,votes[i])
        d.append(temp)
    oo = merge(d)
    #oo = multiply_dicts(oo,label_weights)
    if (gini_dict(oo)<0.4):
        ll = predict_result(oo)
    else:
        ll = predict_average(oo)
    dict_predictions.append(ll)


# In[33]:


dict_predictions


# In[34]:


#Writing the results        
f2 = open("out_boosting_10_classifiers_gini_improved_and_updated_average_with_votes.csv","w")
f2.write("Id,Response\n")
for i in range(0,10000):
    hh = str(20000+i)+","+str(dict_predictions[i])+"\n"
    f2.write(hh) 

f2.close()


# In[35]:


import pickle

f = open('trees_file', 'wb')
pickle.dump(trees, f)
f.close()

f = open('num_trees_file', 'wb')
pickle.dump(num_trees, f)
f.close()

f = open('votes_file', 'wb')
pickle.dump(votes, f)
f.close()

