#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import math
import random
import json
import requests
from matplotlib import pyplot as plt
from pymongo import MongoClient
import copy
from numpy.random import default_rng


# # HYPERPARAMETERS

# In[45]:


NUM_FEATURES=11
LB_LIMIT=-10
UB_LIMIT=10
MAX_ITR=24
MIN_COST=-999999 # ??|
POP_SIZE=40
MUT_RATE=0.3
RETENTION_RATE=0.2  # selection rate from prev generation
TODAY = "[14th March 0.9 x max(t,v) + 0.1 x min(t,v)   SBX RUSSIAN ROULETTE]"


# # CONNECTING TO DATABASE

# In[67]:


client = MongoClient("mongodb+srv://Raj:mongodb123@cluster0.lnsqb.mongodb.net/ml-proj?retryWrites=true&w=majority")
db = client['ml-proj']
coll = db['main-coll']
details = db['details-2']


# In[68]:


surviving_members_num=math.floor(POP_SIZE*RETENTION_RATE)

# SURVIVING MEMBERS COUNT SHOULD BE EVEN
if surviving_members_num%2==1:  
    surviving_members_num-=1
    
num_cells_mutate=math.ceil((POP_SIZE-1)*NUM_FEATURES*MUT_RATE)  # COORDINATES MUTATED EACH ITERATION
num_matings_per_itr=math.ceil((POP_SIZE-surviving_members_num)/2) # MATINGS EACH ITERATION
print("surviving members num is ", surviving_members_num)

vector_info=[] # DETAILS ABOUT EACH VECTOR IN THIS RUN


# In[70]:


API_ENDPOINT = 'http://10.4.21.156'
MAX_DEG = 11
SECRET_KEY = 'BKeUYuWnLNaT7N2zzbmzI9VktYLNGrwbMX2lBfiGksBb3m6m0R'
# SECRET_KEY = "none"

def urljoin(root, path=''):
    if path:
        root = '/'.join([root.rstrip('/'), path.rstrip('/')])
    return root


def send_request(id, vector, path):
    api = urljoin(API_ENDPOINT, path)
    vector = json.dumps(vector)
    response = requests.post(api, data={'id': id, 'vector': vector}).text
    if "reported" in response:
        print(response)
        exit()

    return response

def get_errors(id, vector):
    for i in vector:
        assert 0 <= abs(i) <= 10
    assert len(vector) == MAX_DEG

    return json.loads(send_request(id, vector, 'geterrors'))


# ###  GET TRAINING AND VALIDATION ERRORS FOR VECTOR and making storage arrangements (function used in the code below)

# In[72]:


def get_fitness_value(coeff_arr, curr_gen_num):    
    
    #return [1,2]
    #return random.random()

    assert(len(coeff_arr)==NUM_FEATURES)
    for i in range(0,NUM_FEATURES):
        if coeff_arr[i]>=10:
            coeff_arr[i] = 10
        elif i<=-10:
            coeff_arr[i] = -10

    val = get_errors(SECRET_KEY, coeff_arr)
    #print(f"coeff arr is {coeff_arr}")
    #print("val is ", val)
    vector_info.append({"array_stuff":coeff_arr, "errors_returned":val, "gen":curr_gen_num})
    #print("---------------------------------------------------------------------")
    
    return val


# #### ALTERNATIVE OF ABOVE FOR DEBUGGING PURPOSES

# In[73]:



def get_fitness_value_alt(coeff_arr, curr_gen_num):
    assert(len(coeff_arr)==NUM_FEATURES)
    x=coeff_arr[0]
    y=coeff_arr[1]
    '''x_add=x*math.sin(4*x)
    y_add= 1.1 * y * math.sin(2*y)'''

    '''
    x_add=x*math.sin(4*x)
    y_add= 1.1 * y * math.sin(2*y)
    '''


    # x_add=x
    # y_add=y
    # print(f"x and x_add is {x} : {x_add}")
    # print(f"y and y_add is {y} : {y_add}")
    assert(x>=LB_LIMIT and x<=UB_LIMIT)
    assert(y>=LB_LIMIT and y<=UB_LIMIT)
    # val=x_add+y_add

    # val=0.01*( x*x*(y-2)*(y-3) - (x+y)**3 + y*y*y*(y-2*x))
    val_arr=[x-y]*2
    #print(val)
    #coeff_arr=coeff_arr.tolist()
    vector_info.append({"array_stuff":coeff_arr, "errors_returned":val_arr, "gen":curr_gen_num})
    #res_ret=[val,val]
    #print("val returned is ", res_ret)
    return val_arr


# ### CREATE A STARTING POPULATON FROM NOWHERE (was useful during the very earliest stages mid Feb)

# In[74]:


def create_starting_population(num_members, num_features, lb, ub):
    # Adam and Eve
    sample_pop = np.random.uniform(low=lb, high=ub, size=(num_members,num_features))
    return sample_pop


# In[75]:


# DATABASE OPERATIONS
def insert_single_member_into_db(gen_id, features, errors):
    coll.insert_one({
        'generation': gen_id,
        'try_vector': features,
        'errors': errors,
        'desc' : TODAY
    })
    details.update_one({"try_vector": features},
                           {"$set": {"errors": errors}})


# In[78]:


def calculate_fitness_for_curr_pop(members_list, curr_gen_num):
    '''receives a 2D array'''
    
    tot_len=len(members_list)
    fitness_arr=[]
    
    errors_traced=[]

    for i in range(0,tot_len):
        its_fitness_val = [0, 0]
        exist_vec = coll.find({'try_vector': members_list[i]}).count()

        '''
        For each vector, we search the database if we have seen this vector before to avoid wastage of requests.
        If it is a new vector, it gets added to the database for future reference.
        '''
        print('exist? ', exist_vec)
        if exist_vec > 0:
            doc_found = coll.find_one({'try_vector': members_list[i]})
            its_fitness_val = doc_found['errors']
            details.update_one({"try_vector": members_list[i]},
                           {"$set": {"errors": doc_found['errors']}})
            #print('DOC', doc_found)
            #print('VAL', its_fitness_val)
        else:
            its_fitness_val = get_fitness_value(members_list[i], curr_gen_num)
            insert_single_member_into_db(curr_gen_num, members_list[i],its_fitness_val)
        
        ##########################################################################
        
        
        ''' Here we calculate the fitness function from the training and validation errors
        '''
        
        fitness_val=0
        t=its_fitness_val[0]/(1e10)
        v=its_fitness_val[1]/(1e10)
        fitness_val=0.9 * max(t,v) + 0.1 * min(t,v)
        
        '''if t>1000:
            fitness_val+=1000
            
        if v<10:
            fitness_val-=5'''
            
        
        #####################################################################3
        
        fitness_arr.append(fitness_val)
        
        if type(its_fitness_val)!=list:
            raise TypeError("WEIRD TYPE GOLIATH ERROR")
        errors_traced.append(its_fitness_val)
    #print("fitness arr being sent is ")
    #print(*fitness_arr, sep="\t")
    return fitness_arr,errors_traced


# In[79]:


def sort_generation_members(members_list, curr_gen_num):
    
    #print(f"type is {type(members_list)}  and arg is ")
    #print(*members_list, sep='\n')
    
    tot_len=len(members_list)
    cost_arr , unsorted_errors_traced_here = np.array(calculate_fitness_for_curr_pop(members_list, curr_gen_num))

    # GET THE INDICES OF VECTORS SORTED BY FITNESS (ASCENDING)
    sort_indices = np.argsort(cost_arr)

    # Descending
    # sort_indices=sort_indices[::-1]
    
    if type(members_list[0])!=list:
        raise TypeError("Friday error")
    
    sorted_members_list=[]
    sorted_costs=[]
    # print("member is ", members_list)
    for num in range(0,tot_len):
        i=sort_indices[num]
        #print(f"here member for {i} is ", members_list[i])
        
        sorted_members_list.append(members_list[i])
        sorted_costs.append(cost_arr[i])
        
        
        
        
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        
    
    # print("Sorted indices are \n", sort_indices)
    #sin(x) : 0, 45, 90
    #90, 45,0  (1, 1/root 2, 0)
    #45->1/root 2
    #0->0
    '''print("\nSorted members are \n")
    print(*sorted_members_list, sep='\n')
    print("------------------------------")

    print("\nSorted costs are \n")
    print(*sorted_costs, sep='\n')
    print("#####################################################")
    '''
    return sorted_members_list, sorted_costs, unsorted_errors_traced_here, sort_indices
    


# In[80]:


# Probability of getting selected is higher for a higher rank and vice versa
def get_probabilities(n):
    numerator=np.arange(1,n+1)
    #print("init is ", numerator)
    numerator*=-1
    denom=(n*(n+1))/2
    numerator+=n+1
    #print("new is ", numerator)

    numerator=np.divide(numerator, denom)
    return numerator


# In[81]:


# Probability of getting selected is inversely proportional to fitness
def get_probabilities_russian_roulette(fitness_vals):
    fitness_vals=np.array(fitness_vals)
    print("init is ", fitness_vals)
    numerator=max(fitness_vals) - fitness_vals
    denom=sum(numerator)
 
    numerator=np.divide(numerator, denom)
    print("probabilities being returned is ", numerator)

    return numerator


# In[82]:


# https://stackoverflow.com/a/65671792/6427607
# Stochastic selection

rng = default_rng()
def sus(
        population: np.ndarray,
        fitness: np.ndarray,
        size: int) -> np.ndarray:
    """ https://en.wikipedia.org/wiki/Stochastic_universal_sampling
     """
    if size > len(population):
        raise ValueError

    fitness_cumsum = fitness.cumsum()
    fitness_sum = fitness_cumsum[-1]  # the "roulette wheel"
    step = fitness_sum / size         # we'll move by this amount in the wheel
    start = rng.random() * step       # sample a start point in [0, step)
    # get N evenly-spaced points in the wheel
    selectors = np.arange(start, fitness_sum, step)
    selected = np.searchsorted(fitness_cumsum, selectors)
    #print_report(population, fitness, fitness_cumsum, selectors, selected)
    return selected


# # We tried 3 different approaches for cross overs:
# ## 1) Blend crossover
# ## 2) SBX crossover
# ## 3) Swap crossover (our own logic to handle the affect that ordering of genes can have on single point crossover)
# ## 4) Single Point Crossover
# 

# In[84]:


def blend_crossover(parent1, parent2):

    '''
    [ x1-α(x2-x1), x2+α(x2-x1)] 
    '''
    child1 = np.empty(NUM_FEATURES)
    child2 = np.empty(NUM_FEATURES)
    #print(f"parent 1 is {parent1}")
    #print(f"parent 2 is {parent2}")
    
    num_genes=len(parent1)
    alpha=0.5

    
    for i in range(num_genes):
        max_val=max(parent1[i], parent2[i])
        min_val=min(parent1[i], parent2[i])
        
        lb=min_val-alpha*(max_val-min_val)
        ub=max_val+alpha*(max_val-min_val)
        
        set_val_1=random.uniform(lb, ub)
        child1[i]=set_val_1
        
        set_val_2=random.uniform(lb, ub)        
        child2[i]=set_val_2      
        
    
    
    for i in range(len(child1)):
        child1[i]=max(child1[i],LB_LIMIT)
        child1[i]=min(child1[i],UB_LIMIT)
        
    for i in range(len(child2)):
        child2[i]=max(child2[i],LB_LIMIT)
        child2[i]=min(child2[i],UB_LIMIT)
    
    #print(f"CHILD 1 is {child1}")
    #print(f"CHILD 2 is {child2}")
    
    print("#---------------------------------------")

    return child1.tolist(), child2.tolist()


# In[85]:


def binary_crossover(parent1, parent2):

    '''
    Deb and Agrawal [196] developed the simulated binary crossover (SBX) to simulate the
    behavior of the one-point crossover operator for binary representations. 
    '''
    child1 = np.empty(NUM_FEATURES)
    child2 = np.empty(NUM_FEATURES)
    #print(f"parent 1 is {parent1}")
    #print(f"parent 2 is {parent2}")
    

    u = random.random() 
    n_c = 2
    '''where rj ∼ U(0, 1), and η > 0 is the distribution index. Deb and Agrawal suggested
    that η = 1.'''
    
    '''The SBX operator generates offspring symmetrically about the parents, which prevents
    bias towards any of the parents. For large values of η there is a higher probability that
    offspring will be created near the parents. For small η values, offspring will be more
    distant from the parents.'''
        
    if (u < 0.5):
        beta = (2 * u)**((n_c + 1)**-1)
    else:
        beta = ((2*(1-u))**-1)**((n_c + 1)**-1)


    parent1 = np.array(parent1)
    parent2 = np.array(parent2)
    child1 = 0.5*((1 + beta) * parent1 + (1 - beta) * parent2)
    child2 = 0.5*((1 - beta) * parent1 + (1 + beta) * parent2)
    
    for i in range(len(child1)):
        child1[i]=max(child1[i],LB_LIMIT)
        child1[i]=min(child1[i],UB_LIMIT)
        
    for i in range(len(child2)):
        child2[i]=max(child2[i],LB_LIMIT)
        child2[i]=min(child2[i],UB_LIMIT)
        
    #child1=(parent1+parent2)/2
    #child2=(parent1-parent2)/2
    
    #print(f"CHILD 1 is {child1}")
    #print(f"CHILD 2 is {child2}")
    
    #print("#---------------------------------------")

    return child1.tolist(), child2.tolist()


# In[7]:


def swap_crossover(parent1, parent2):
    
#     NUM_FEATURES = 5
    child1 = np.empty(NUM_FEATURES)
    child2 = np.empty(NUM_FEATURES)
    #print(f"parent 1 is {parent1}")
    #print(f"parent 2 is {parent2}")
    
    num_genes=len(parent1)
    
    ## child-1 parent 1 feaures =4
    ## child-2 parent 1 feaures =7
    
    ## child-1 parent 2 feaures =7
    ## child-2 parent 2 feaures =4    
    
    #a1, a2, ,,,a11
    #b1, b2, ,,,b11
    
    #a1, b2, b3, a4
    #b1, a2, a3, b4
    
    # Number of distinct random indices at which swapping occurs in the children vectors.
    num_swap=3
    swap_indices = random.sample(range(0,num_genes), num_swap)
    print(swap_indices)

    
    for i in range(num_genes):
        if i in swap_indices:
            child1[i] = parent1[i]
            child2[i] = parent2[i]
        else:
            child1[i] = parent2[i]
            child2[i] = parent1[i]
        
    
    for i in range(len(child1)):
        child1[i]=max(child1[i],LB_LIMIT)
        child1[i]=min(child1[i],UB_LIMIT)
        
    for i in range(len(child2)):
        child2[i]=max(child2[i],LB_LIMIT)
        child2[i]=min(child2[i],UB_LIMIT)
    
    #print(f"CHILD 1 is {child1}")
    #print(f"CHILD 2 is {child2}")
    
    print("#---------------------------------------")

 

    return child1.tolist(), child2.tolist()


# ### We experimented with simulated annealing to handle mutation magnitude but did not find any good results on using it

# In[98]:


'''
This is a type of mutation which gradually decreases the magnitude of mutation as we move to higher generations
'''

def simulated_annealing(val, temp):
    tou = random.randrange(-1,2,2) # increase or decrease
    r = random.uniform(0.4,0.6)     # random number
    b = 3                         # design parameter
    print('r',r)
    print('design paramter',b)
    print('tou',tou)
    
    
    new_val = val + tou*(UB_LIMIT - LB_LIMIT)*(1 - r**((1 - temp/MAX_ITR)**b))
    
    
    return new_val


# In[193]:


simulated_annealing(3, 20)


# ### When we didnt find enough vectors for our starting population, we mutated the ones we had to fill up for the rest.

# In[88]:


def get_mutated_version(arr, really=False):
    mut_prob_now=0.7
    arr=arr.tolist().copy()
    if really:
        #arr=arr.tolist()
        #print(arr)        
        len_arr=len(arr)
        for i in range(0,len_arr):
            prob=random.uniform(0, 1)
            if prob<=mut_prob_now:
                curr_val=arr[i]
                # in order to mutate 0, we set it to 1
                if curr_val==0:
                    curr_val=1
                d_val=random.uniform(-abs(4*curr_val), abs(4*curr_val))
                arr[i]+=d_val
                arr[i]=min(arr[i],UB_LIMIT)
                arr[i]=max(arr[i],LB_LIMIT)
                #print(f"{i} mutated {curr_val}->{arr[i]}")
        return arr
    else:
        return arr
        


# ### Initializing the initial population based on the few best vectors we had over the past days ordered on basis of  different fitness functions (details in REPORT) and keeping in mind that initial members of the population must be diverse (having similar vectors would not cause any gain)

# In[89]:


# Iterations


CURR_GEN_NUM=1

init_arr = np.array(
[
    [0.1404353012090963, -1.5917637608939356e-12, -1.2461542166753883e-13, 6.945508820550938e-11, -2.833820715681869e-10, -4.995728196797225e-16, 6.007192185810843e-16, 2.3151128263733076e-05, -1.5030814308516114e-06, -1.3260366172169755e-08, 6.710158549311463e-10],
    [0.07340107154031256, -1.5437615866430792e-12, -1.4641151122621188e-13, 6.645245422169678e-11, -3.7164454801951106e-10, -3.6675104747060576e-16, 4.671199506291923e-16, 2.37846748093054e-05, -1.4689166025106545e-06, -1.4156223932628392e-08, 6.63936470940267e-10],
    [0.03523998918426694, -1.0117081002184981e-12, -1.2994472471499564e-14, 7.893352705113582e-11, -1.1860894152791612e-10, -1.485052578582483e-15, 1.3056718502199423e-15, 2.0174164573336113e-05, -1.812224643455608e-06, -8.514787778320894e-09, 7.44044375654244e-10], 
    [0.10456903003668358, -8.310365465172867e-13, -1.1425053328907633e-14, 9.427139040575664e-11, -1.313228713688508e-10, -1.1727056219971407e-15, 1.0779150203669628e-15, 1.643454994947712e-05, -1.6601745706684455e-06, -6.180195958180512e-09, 6.592835767247373e-10],
    [-0.15312064598796957, -1.2759516587457305e-12, -6.334633169412846e-14, 1.0791176309199842e-10, -4.5155117540841926e-10, -1.6054819562735365e-15, 8.06656115843023e-16, 3.113179944203683e-05, -1.9525844629653752e-06, -1.1537654091036487e-08, 7.67663074742261e-10],
    [0.05271366053724377, -1.6029419079568322e-12, -1.1478915535579567e-13, 7.996355167326429e-11, -1.9736060467401116e-10, -2.5951233583070186e-16, 6.595567762286758e-16, 2.6845001767140418e-05, -1.2827130152295797e-06, -1.939661532122104e-08, 6.55003088671148e-10]
])

#print(init_arr.shape)

#init_arr=np.array([LB_LIMIT+0.1,UB_LIMIT-0.1])

## Create the initial population
starting_population_mat=create_starting_population(POP_SIZE, NUM_FEATURES, LB_LIMIT, UB_LIMIT)

for i in range(0,POP_SIZE):
    if i<6:
        starting_population_mat[i]=get_mutated_version(np.array(init_arr[i%6]), False)
    else:
        starting_population_mat[i]=get_mutated_version(np.array(init_arr[i%6]),True)
        
'''starting_population_mat = np.array([                                    
])
   '''

starting_population_mat=starting_population_mat
print("\nStarting pop is \n")
print(*starting_population_mat, sep="\n\n")

generation_info=[]
generation_info.append({"members":starting_population_mat.tolist(), "avg_fitness":0, "min_fitness":0})
#print(generation_info)





# In[24]:


TODAY


# ### Making arrangements to store the results returned by the valuable queries on the server

# In[90]:


# Information insert into database about starting population
for member in starting_population_mat:
    member=member.tolist()
    details.insert_one({
        'generation': CURR_GEN_NUM,
        'try_vector': member, #after mutation
        'errors': [-1,-1],
        'mother_vec': member,
        'father_vec': member,
        'before_mutation': member,
        'beta': -1,
        'cross_over_point': -1,
        'desc' : TODAY
        })


# In[91]:


trace_arr=[]
trace_now=[]

for i in range(0,POP_SIZE):
    vec=copy.deepcopy(starting_population_mat[i])
    vec=vec.tolist()
    obj_to_store={
        'generation': CURR_GEN_NUM,
        'try_vector': vec, #after mutation
        'errors': [-1,-1],
        'mother_vec':vec,
        'father_vec': vec,
        'before_mutation':vec,
        'beta': -1,
        'cross_over_point': -1,
        'desc' : TODAY, 
        'survived':False
        }
    trace_now.append(obj_to_store)

print(*trace_now, sep='\n\n')


# #### Sanity checking confirmation statements to prevent overwriting, manipulation of parameters etc
# 

# In[92]:


'''Safety purposes to avoid unwanted glitches'''

verified=input("Have you taken care of get_fitness_vals, \nexist_vec!=0 , \ncollection name,\ndetails name ?\nHave you switched on VPN ?\n JSON file name\n\nHAVE YOU CHANGED METRIC IN FITNESS VALUE APPEND FUNCTION ?")
if verified!='y':
    raise TypeError    


# In[93]:


MUT_RATE


# ## THE MAIN PROCESS

# In[94]:


mut_magnitude=0.2
# mutation magnitude is the amount by which the values of attributes
# get altered in a vector, if they are changed


tot_crosses=0
same_crosses=0


if type(starting_population_mat)!=np.ndarray:
    raise TypeError("Mismatch of starting pop mat")
    
    
for itr_num in range(0, MAX_ITR+1):
    #print("args being sent are \n")
    #print(*generation_info[-1]["members"],sep='\n')
    parents_now, sorted_fitness_vals, unsorted_errors_main, sorted_indices = sort_generation_members(generation_info[-1]["members"], CURR_GEN_NUM)
    
    for i in range(POP_SIZE):
        trace_now[i]["errors"]=unsorted_errors_main[i]

        '''if unsorted_errors_main[i]!=list:
            raise TypeError("Pitts ERROR")'''
        
    trace_tmp=copy.deepcopy(trace_now)
    
    for i in range(0,POP_SIZE):
        trace_now[i]=copy.deepcopy(trace_tmp[sorted_indices[i]])
        
    trace_arr.append(copy.deepcopy(trace_now))
    
    
    #[{obj}] -> obj -> errors -> fitness value
    
    
    # Recording information of previous generation
    generation_info[-1]["avg_fitness"]=np.average(sorted_fitness_vals)
    generation_info[-1]["min_fitness"]=np.min(sorted_fitness_vals)
    generation_info[-1]["gen_number"]=CURR_GEN_NUM
    generation_info[-1]["fitness_vals_stored"]=sorted_fitness_vals
    print(f'Avg fitness is {generation_info[-1]["avg_fitness"]}' )
    print(f'MIN fitness is {generation_info[-1]["min_fitness"]}' )
    print(f"fitness vals are: -> \n")
    print(*sorted_fitness_vals, sep='\n')

    
    # We put a check after each iteration so we could stop in between
    #stop_or_not = input("Enter y to continue.")
    stop_or_not = 'y'
    if stop_or_not!='y' or itr_num==MAX_ITR:
        break
        
    ##################################################################################
    
    CURR_GEN_NUM+=1
    # https://stackoverflow.com/a/7851237/6427607
    matings_pending=num_matings_per_itr
    probs=get_probabilities_russian_roulette(sorted_fitness_vals)
    cum_probs=np.cumsum(probs)
    # https://numpy.org/doc/stable/reference/generated/numpy.hstack.html
    cum_probs=np.hstack(([0],cum_probs))
    #print(f"Probs are {probs}\nCumProbs are {cum_probs}")


    '''Selection of parents who will take part in crossover
    '''
    pick_1=np.random.uniform(low=0, high=1, size=(matings_pending))
    pick_2=np.random.uniform(low=0, high=1, size=(matings_pending))

    mother_idx=[]
    father_idx=[]
    
    ## Making the parents breed

    for ic in range(0,matings_pending):
        tot_crosses+=1
        for idx in range(1,POP_SIZE+1):
            if pick_1[ic]<=cum_probs[idx] and pick_1[ic]>cum_probs[idx-1]:
                mother_idx.append(idx-1)
            if pick_2[ic]<=cum_probs[idx] and pick_2[ic]>cum_probs[idx-1]:
                father_idx.append(idx-1)
        # avoid homogeneous crossover
        if mother_idx[-1]==father_idx[-1]:
                father_idx[-1]=((mother_idx[-1])+1)%POP_SIZE
    

    new_parents=parents_now.copy()   
    for i in range(0, surviving_members_num):
        details.insert_one({
        'generation': CURR_GEN_NUM,
        'try_vector': new_parents[i], #after mutation
        'errors': [-1,-1],
        'mother_vec': new_parents[i],
        'father_vec': new_parents[i],
        'before_mutation': new_parents[i],
        #'beta': -1,
        #'cross_over_point': -1,
        'desc' : TODAY
        })
    #trace_now=copy.deepcopy(trace_prev)

    for mating_num in range(0, matings_pending):
        par_one_idx=mother_idx[mating_num]
        par_two_idx=father_idx[mating_num]

        child_one=parents_now[par_one_idx].copy()
        child_two=parents_now[par_two_idx].copy()
        
        
        '''We used to change the crossover function here to serve our purpose'''
        child_one, child_two=binary_crossover(child_one, child_two)

    
        '''CrossOver
        '''
        '''special_cell_idx=random.randint(0,NUM_FEATURES-1)

        beta=random.random()

        xy=parents_now[par_one_idx][special_cell_idx]-\
            parents_now[par_two_idx][special_cell_idx]

        child_one[special_cell_idx]=parents_now[par_one_idx][special_cell_idx] - beta*xy
        child_two[special_cell_idx]=parents_now[par_two_idx][special_cell_idx] + beta*xy


        if special_cell_idx!=NUM_FEATURES-1:           
            #print("CHILD ONE IS ", child_one)
            #print("CHILD TWO IS ", child_two)
            child_one[:] = np.hstack((child_one[:special_cell_idx]   \
                            ,   child_two[special_cell_idx:]    ))      

            child_two[:] = np.hstack((child_two[:special_cell_idx]   \
                            ,   child_one[special_cell_idx:]    ))   '''
            
        
        '''Vectors which rank at the top of the generation are retained while those below are overwritten
        by new children.
        '''
        
        new_parents[surviving_members_num + 2* mating_num] = child_one
        new_parents[surviving_members_num + 2* mating_num + 1] = child_two
        
        if type(child_one)!=list:
            raise TypeError("NOT A LIST")
        
        if type(parents_now[par_one_idx])!=list:
            raise TypeError("IT ALSO NOT A LIST")
            
        obj_one={
        'generation': CURR_GEN_NUM,
        'try_vector': child_one, #after mutation
        'errors': [-1,-1],
        'mother_vec':parents_now[par_one_idx],
        'father_vec': parents_now[par_two_idx],
        'before_mutation': child_one,
        #'beta': beta,
        #'cross_over_point': special_cell_idx,
        'desc' : TODAY, 
        'survived':False, 
        "mut_rate":MUT_RATE
        }
        
        obj_two={
        'generation': CURR_GEN_NUM,
        'try_vector': child_two, #after mutation
        'errors': [-1,-1],
        'mother_vec':parents_now[par_one_idx],
        'father_vec': parents_now[par_two_idx],
        'before_mutation': child_two,
        #'beta': beta,
       # 'cross_over_point': special_cell_idx,
        'desc' : TODAY, 
        'survived':False, 
        "mut_rate":MUT_RATE
        
        }
        
        trace_now[surviving_members_num + 2* mating_num]=obj_one
        trace_now[surviving_members_num + 2* mating_num + 1] =obj_two
        details.insert_one(copy.deepcopy(obj_one))
        details.insert_one(copy.deepcopy(obj_two))


    '''
    %_______________________________________________________
    % Mutate the population
    '''

    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html
    mut_rows=np.random.randint(POP_SIZE, size=num_cells_mutate)
    for i in range(0,len(mut_rows)):
        # mut_rows[i]=int(math.round(mut_rows[i]))
        assert(mut_rows[i]<POP_SIZE)
    mut_rows.sort()

    
    # mut_cols=np.random.uniform(low=0, high=1, size=(num_cells_mutate))
    # mut_cols*=(NUM_FEATURES-1)

    mut_cols=np.random.randint(NUM_FEATURES, size=num_cells_mutate)

    for i in range(0,len(mut_cols)):
        #mut_cols[i]=int(math.round(mut_cols[i]))
        assert(mut_cols[i]<NUM_FEATURES)

    for i in range(0,num_cells_mutate):
        row_c=int(mut_rows[i])
        col_c=int(mut_cols[i])
  
        
        upd_val = simulated_annealing(new_parents[row_c][col_c], CURR_GEN_NUM)
        
        
        upd_val = max(LB_LIMIT, upd_val)
        upd_val = min(UB_LIMIT, upd_val)
        
        
        old_vec= copy.deepcopy(new_parents[row_c])
        new_parents[row_c][col_c] = upd_val
        details.update_one({"before_mutation": old_vec},
                           {"$set": {"try_vector": new_parents[row_c]}})
        
        

        
        
    
    
    ##///////////////////////////////////////////////////////////////////////////////////
    
    for i in range(POP_SIZE):
        if type(new_parents[i])!=list:
            raise TypeError("Pegasus type error")
        #trace_now[i]["try_vector"]=new_parents[i]
        trace_now[i]["try_vector"]=copy.deepcopy(new_parents[i])
        if i< surviving_members_num:
            trace_now[i]["survived"]=True
    ##///////////////////////////////////////////////////////////////////////////////////
    
    
    #print("Generation info is ", generation_info)
    
    #############################################################################
    
    # PLOTTING OF GENERATION VS FITNESS to get an idea about how things are proceeding with each generation
    x_c=[]
    y_min=[]
    y_avg=[]
    #print(len(generation_info))
    for j in range(0,len(generation_info)):
        x_c.append(j)
        y_min.append(generation_info[j]["min_fitness"])
        y_avg.append(generation_info[j]["avg_fitness"])

    plt.figure(figsize=(8,8))
    plt.plot(x_c, y_min, label="min")
    plt.show()
    plt.figure(figsize=(8,8))
    plt.plot(x_c, y_avg, label="avg")
    #print('Min: ',y_min[-1])
    #print('Avg: ',y_avg[-1])
    #print('Vector: ',generation_info[-1]["members"])
    #print(*y_min,sep='\n')
    plt.legend()
    plt.show()
    
    
    print("--------------------------------------------------------------")
    
    '''
    %_______________________________________________________
    % The new offspring and mutated chromosomes are
    % evaluated
    '''
    #print(type(new_parents))
    
    # This generation becomes the new parents for the next generation
    parents_now=new_parents
    generation_info.append({"members":new_parents, "mut_magnitude":mut_magnitude})

    ''' We kept mutation magnitude and the fraction of cells to be mutated as flexible so that we can manipulate them in case we get bad/unrealistic results
    '''
    
    mut_magnitude=float(input("Enter mutation magnitude:"))
    MUT_RATE=float(input("Enter fraction of cells you want to mutate:"))
    num_cells_mutate=math.ceil((POP_SIZE-1)*NUM_FEATURES*MUT_RATE) ## ??????


# In[95]:


print(tot_crosses)
print(same_crosses)


# In[96]:


print(len(generation_info))


# In[97]:


generation_info[0]


# In[33]:



#print(get_fitness_value(generation_info[0]["members"][0]))


# In[98]:


print(vector_info)


# In[99]:


# Dumping all data into json
with open('filename_DAY_STINT_NUMBER.json','w') as fd:
    obj={}
    print(type(vector_info[0]["errors_returned"]))
    obj["anmol_arr"]=vector_info
    obj["generation_info"]=generation_info
    obj["trace_arr"]=trace_arr
    #print(type(vector_info))
    #obj["b"]=[1,2,3,4]
    json.dump(obj, fd)


# In[100]:


print("trace_arr is ", trace_arr)


# In[ ]:





# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=ac87f5b3-e165-4c1b-9ae8-00db3f497180' target="_blank">
# <img style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
