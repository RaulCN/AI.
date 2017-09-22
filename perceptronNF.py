##Returns the new decision tree based on the examples given
#@param data: Copy of the data list.
#@param attributes: List of all attributes 
#@param target_attr: The targer attribute that will be the evaluated
#@param fitness_func: The target function
def create_decision_tree(data,attributes,target_attr,fitness_func):
    
    data = data[:]
    vals = [record[target_attr] for record in data]
    default = majority_value(data, target_attr)
    
    #if the dataset is empty or the attributes list is empty, return the
    #the default value. When checking the attributes list for emptiness,
    #we need to subtract 1 to account for the target attribute.
    if not data or (len(attributes)-1) <= 0:
        return default
    #if all the records in the dataset have the same classification,
    #return the classification.
    elif vals.count(vals[0]) == len(vals):
        return vals[0]
    
    else:
        #Choose the next best attribute to best classify our data
        best = choose_attribute(data,attributes,target_attr,fitness_func)
        
        #Create a new decision tree/node with the best attribute.
        tree = {best:{}}
        
        #Create a new decision tree/sub-node for each of the values in the
        #best attribute field
        for val in get_values(data,best):
            #create a subtree for the current value under the "best" field
            subtree = create_decision_tree(get_examples(data,best,val),
                [attr for attr in attributes if attr != best],
                target_attr,
                fitness_func)
        
            # Add the new subtree to the empty dictionary object in our new
            # tree/node we just created.
            tree[best][val] = subtree
        
    return tree

##Calculates the entropy of the given data set for the target attribute.
#@param data: the data list
#@param target_attr = the target attribute
def entropy(data,target_attr):
    val_freq = {}
    data_entropy = 0.0
    
    #Calculates the frequency of each of the values in the target attribute.
    for record in data:
        if val_freq.has_key(record[target_attr]):
            val_freq[record[target_attr]] += 1.0
        else:
            val_freq[record[target_attr]] = 1.0
    
    #Calculate the entropy of the data for the target attribute
    for freq in val_freq.values():
            data_entropy += (-freq/len(data)) * math.log(freq/len(data),2)
    

    return data_entropy

##Calculates the information gain (reduction in entropy) that
##would result by splitting the data on the chosen attribute (attr).
def gain(data, attr, target_attr):
    val_freq = {}
    subset_entropy = 0.0
    
    #Calculate the frequency of each of the value in the target attribute
    for record in data:
        if val_freq.has_key(record[attr]):
            val_freq[record[attr]] += 1.0
        else:
            val_freq[record[attr]] = 1.0
    
    #Calculate the sum of the entropy for each subset of records weighted
    #by their probability of ocurring in the training set.
    for val in val_freq.keys():
        val_prob = val_freq[val] / sum(val_freq.values())
        data_subset = [record for record in data if record[attr]==val]
        subset_entropy += val_prob * entropy(data_subset,target_attr)
 
    #subtract the entropy of the chosen attribute from the entropy of the
    #whole data set with respect to the target attribute (and return it)
    return (entropy(data,target_attr) - sub

