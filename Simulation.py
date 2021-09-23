import igraph
#import numba
#from numba import cuda
#from numba import jit
import Network
import time
import numpy as np
import math
import plotly.express as px
import matplotlib.pyplot as plt
from pprint import pprint

'''
arrays:
    infected
    susceptible
    contagiousprint

    adjacency matrix - np [n][n]
'''


#@jit(nopython=True)
def get_contagious_contacts(adjacency_matrix, susceptible_nodes, contagious_nodes, infection_rate=1):
    contagious_contacts = (adjacency_matrix.dot(contagious_nodes)).T[0] * susceptible_nodes * infection_rate
    #print(sum(contagious_contacts))
    return contagious_contacts


def infication_roulette_wheel_choise(contagious_contacts, contagious_contacts_concentration):
    probabilities = contagious_contacts / contagious_contacts_concentration
    if(contagious_contacts_concentration <= 0):
        print("ERROR: Contagious_contacts_concentration <= 0")
        return -1
    
    sumprob = 0.0
    inficate = np.random.uniform(0.0, 1.0) # [0, 1)
    for i in range(len(contagious_contacts)):
        previous_sumprob = sumprob
        sumprob += probabilities[i]
        if(previous_sumprob <= inficate and inficate < sumprob):
            return i


def get_time_to_infication(contagious_contacts_concentration):
    #check with R rexp(1,qii)
    if(contagious_contacts_concentration <= 0):
        return math.inf
    return round(np.random.exponential(1/contagious_contacts_concentration),12) # expected value - contagious_contacts_concentration.
    #return round(1/contagious_contacts_concentration)


def infication(index_node_for_infication, infication_time, infected_nodes, susceptible_nodes, contagious_nodes, times_node_infication, death_note):
    prob_of_death = 0.01820518643617639 + 0.2
    infected_nodes[index_node_for_infication] = 1
    susceptible_nodes[index_node_for_infication] = 0
    contagious_nodes[index_node_for_infication][0] = 1
    times_node_infication[index_node_for_infication] = infication_time
    if(np.random.uniform(0.0, 1.0) <= prob_of_death):
        death_note[index_node_for_infication] = 1 #die with probability


def CTMC(adjacency_matrix, susceptible_nodes, infected_nodes, contagious_nodes, times_node_infication, death_note, treatment_time, critically_treatment_time, infection_rate = 0.01, time = 0):
    # do actions (threatment, death, etc)
    do_actions(time, infected_nodes, susceptible_nodes, contagious_nodes, times_node_infication, death_note, treatment_time, critically_treatment_time)
    contagious_contacts = get_contagious_contacts(adjacency_matrix, susceptible_nodes, contagious_nodes, infection_rate)
    contagious_contacts_concentration = sum(contagious_contacts)
    if(contagious_contacts_concentration <= 0):
        # what can i do?
        pass
    index_node_for_infication = infication_roulette_wheel_choise(contagious_contacts,contagious_contacts_concentration)
    infication(index_node_for_infication, time, infected_nodes, susceptible_nodes, contagious_nodes, times_node_infication, death_note)
    return get_time_to_infication(contagious_contacts_concentration)


def start_infication(number_of_infications, infected_nodes, susceptible_nodes, contagious_nodes, times_node_infication, death_note):
    infication_time = 0
    for _ in range(number_of_infications):
        index_node_for_infication = np.random.randint(0, len(infected_nodes))
        infication(index_node_for_infication, infication_time, infected_nodes, susceptible_nodes, contagious_nodes, times_node_infication, death_note)


def get_time_to_action(current_time, time_to_next_infication, time_step):
    new_time = min(current_time + time_to_next_infication, current_time + time_step - current_time % time_step)
    return round(new_time, 12), round(time_to_next_infication -  (new_time - current_time), 12)


def do_actions(time, infected_nodes, susceptible_nodes, contagious_nodes, times_node_infication, death_note, treatment_time, critically_treatment_time):
    for i in range(len(infected_nodes)):
        if(infected_nodes[i] == 1 and death_note[i] == 1 and time >= times_node_infication[i]+critically_treatment_time):
            infected_nodes[i] = 0
            susceptible_nodes[i] = 0
            contagious_nodes[i] = 0
        if(infected_nodes[i] == 1 and death_note[i] == 0 and time >= times_node_infication[i]+treatment_time):
            infected_nodes[i] = 0
            susceptible_nodes[i] = 0
            contagious_nodes[i] = 0
    #print("infected_nodes: ", infected_nodes)
    #print(treatment_time)
            

def get_states_info(infected_nodes, susceptible_nodes, contagious_nodes, death_note):
    amount_of_infected = sum(infected_nodes)
    amount_of_susceptible = sum(susceptible_nodes)
    amount_of_contagious = sum(contagious_nodes.T[0])
    amount_of_critically_infected = sum([1 for have_to_die, infected in zip(death_note, infected_nodes) if(have_to_die == 1 and infected == 1)])
    amount_of_dead = sum([1 for have_to_die, infected in zip(death_note, infected_nodes) if(have_to_die == 1 and infected == 0)])
    return amount_of_infected, amount_of_susceptible, amount_of_contagious, amount_of_critically_infected, amount_of_dead


def simulation(graph_size, network_type, amount_of_contacts, infection_rate, number_of_infications, max_time, time_step, i):
    infected_nodes = [0 for i in range(graph_size)]
    susceptible_nodes = [1 for i in range(graph_size)]
    contagious_nodes = np.array([[0] for i in range(graph_size)])
    times_node_infication = [-1] * graph_size

    death_note = [0 for i in range(graph_size)]
    #print("death note", death_note)

    treatment_time = 10
    critically_treatment_time = 14

    all_time = time.time()
    graph = Network.create_network(graph_size, amount_of_contacts, network_type)
    adjacency_matrix = np.array(list(graph.get_adjacency()))
    start_infication(number_of_infications, infected_nodes, susceptible_nodes, contagious_nodes, times_node_infication, death_note)
    time_to_next_infication = get_time_to_infication(np.sum(get_contagious_contacts(adjacency_matrix,susceptible_nodes, contagious_nodes, infection_rate)))
    current_time = 0
    times_to_infications = [time_to_next_infication]

    states_info = [["time", "amount_of_infected", "amount_of_susceptible", "amount_of_contagious", "amount_of_critically_infected", "amount_of_dead"]]
    
    #print("current_time: ", current_time)

    while(current_time < max_time):
        current_time, time_to_next_infication = get_time_to_action(current_time, time_to_next_infication, time_step)
        #print("current_time: ", current_time, "\t time to next infication: ", time_to_next_infication)
        if(time_to_next_infication == 0):
            time_to_next_infication = CTMC(adjacency_matrix, susceptible_nodes, infected_nodes, contagious_nodes, times_node_infication, death_note, treatment_time, critically_treatment_time, infection_rate, current_time)
            times_to_infications.append(time_to_next_infication)
            #print("new time_to_next_infication: ", time_to_next_infication)
        if(current_time % time_step == 0):
            do_actions(current_time, infected_nodes, susceptible_nodes, contagious_nodes, times_node_infication, death_note, treatment_time, critically_treatment_time)
            states_info.append([current_time] + list(get_states_info(infected_nodes, susceptible_nodes, contagious_nodes, death_note)))
            
    #print(times_to_infications)
    #plt = px.line(x=range(len(times_to_infications)),y=times_to_infications).show() 
    #plt.plot(range(len(times_to_infications)), times_to_infications)
    #plt.show()

    #pprint(states_info)

    print("all time: ", time.time() - all_time)
    with open('/media/user/5e05c37c-4b21-4166-aedf-ce93f230063f/projects/ManagingEpidemicOutbreak/Python-v.0.1/simulations/' + str(i) + '.txt', 'w') as file:
        for row in states_info:
            file.write(','.join([str(a) for a in row]) + '\n')
    

graph_size = (10 ** 2) * 2
network_type = 'Complete'
amount_of_contacts = 0
infection_rate = 0.002
number_of_infications = 1
max_time = 100
time_step = 1

for i in range(0, 150):
    simulation(graph_size, network_type, amount_of_contacts, infection_rate, number_of_infications, max_time, time_step, i)



# def test_simulation(graph_size, network_type, amount_of_contacts, infection_rate, number_of_infications, max_time, time_step):
#     infected_nodes = [0 for i in range(graph_size)]
#     susceptible_nodes = [1 for i in range(graph_size)]
#     contagious_nodes = [[0] for i in range(graph_size)]
#     times_node_infication = [-1] * graph_size


#     all_time = time.time()
#     start = time.time()
#     graph = Network.create_network(graph_size, amount_of_contacts, network_type)
#     print("time of network creation: ", time.time() - start)

#     print('two')
#     start = time.time()
#     adjacency_matrix = np.array(list(graph.get_adjacency()))
#     print("time of getting adjecency matrix: ", time.time() - start)

#     print('three')
#     start = time.time()
#     start_infication(number_of_infications, infected_nodes, susceptible_nodes, contagious_nodes, times_node_infication)
#     print("time of start infication: ", time.time() - start)

#     print('four')
#     start = time.time()
#     time_to_next_infication = get_time_to_infication(np.sum(get_contagious_contacts(adjacency_matrix,susceptible_nodes, contagious_nodes, infection_rate)))
#     print("time of getting time: ", time.time() - start)
#     current_time = 0
#     print("current_time: ", current_time)


#     times_to_infications = [time_to_next_infication]

#     while(current_time < max_time):
#         current_time, time_to_next_infication = get_action_time(current_time, time_to_next_infication, time_step)
#         #print("current_time: ", current_time, "\t time to next infication: ", time_to_next_infication)
#         #time.sleep(0.1)
#         if(time_to_next_infication == 0):
#             time_to_next_infication = CTMC(adjacency_matrix, susceptible_nodes, infected_nodes, contagious_nodes, times_node_infication, infection_rate, current_time)
#             times_to_infications.append(time_to_next_infication)
#             #print("new time_to_next_infication: ", time_to_next_infication)
#         if(current_time % time_step == 0):
#             #print(sum(infected_nodes))
#             pass
#     print(sum(infected_nodes))
#     #print(infected_nodes)
#     print(times_to_infications)
#     plt = px.line(x=range(len(times_to_infications)),y=times_to_infications).show() 
#     #plt.plot(times_to_infications, range(len(times_to_infications)))

#     print("all time: ", time.time() - all_time)


# graph_size = 10**4
# amount_of_contacts = 5

# matrix = np.array(list(graph.get_adjacency()))
# #print(get_inficated_contacts(np.array(graph.get_adjacency()), graph.vcount(), 0))
# #print(matrix)
# print(len(matrix))
# test_inficated_arr = np.random.randint(0,2, size=graph_size)
# # matrix = cuda.to_device(matrix)
# # test_inficated_arr = cuda.to_device(test_inficated_arr)
# start_time = time.time()
# get_inficated_contacts(matrix, len(matrix), test_inficated_arr)
# print("--- %s seconds ---" % (time.time() - start_time))
# #print(graph)
# #adjacency_matrix = graph.get_adjacency()
# #print(adjacency_matrix)
# # print(type(adjacency_matrix))