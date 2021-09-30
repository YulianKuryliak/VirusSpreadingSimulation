import igraph
#import numba
#from numba import cuda
#from numba import jit
import Network_model
import Population 
import time
import numpy as np
import math
#import plotly.express as px
import matplotlib.pyplot as plt
from pprint import pprint
import os
from os import path

'''
arrays:
    infected
    susceptible
    contagiousprint

    adjacency matrix - np [n][n]
'''

#population_data {"age_range", "gender", "probability", "cumulative"}

class Network:

    def __init__(self, adjacency_matrix, population_data):
        self.adjacency_matrix = adjacency_matrix
        self.size = len(adjacency_matrix)
        self.__init_disease_info()
        self.__init_population_info(population_data)


    def __init_disease_info(self):
        self.infected_nodes = [0 for i in range(self.size)]
        self.susceptible_nodes = [1 for i in range(self.size)]
        self.contagious_nodes = np.array([[0] for i in range(self.size)])
        self.times_node_infication = [-1] * self.size
        self.ages = [-1 for i in range(self.size)]
        self.genders = ['U' for i in range(self.size)]
        self.death_probabilities = [0 for i in range(self.size)]


    def __init_population_info(self, data):
        for i in range(self.size):
            self.ages[i], self.genders[i], self.death_probabilities[i] = self.__searching_data(data, np.random.uniform(0.0, 1.0))
            

    def __searching_data(self, data, probability):
        for i in range(1,len(data)):
            if(probability < data[i]["cumulative"]):
                return data[i]["age_range"], data[i]["gender"], data[i]["p_death"]
        #return data[len(data)-1]["age_range"], data[len(data)-1]["gender"]

    def print(self):
        for i in range(self.size):
            print("node: %5d, age: %7s, gender: %2c, infected: %2d, susceprible: %2d, contagious: %2d, time_node_infication: %6f, prbability of death: %6f" % (
                i, self.ages[i], self.genders[i], self.infected_nodes[i], self.susceptible_nodes[i], self.contagious_nodes[i][0], self.times_node_infication[i], self.death_probabilities[i]))


#@jit(nopython=True)
def get_contagious_contacts(network, infection_rate=1):
    contagious_contacts = (network.adjacency_matrix.dot(network.contagious_nodes)).T[0] * network.susceptible_nodes * infection_rate
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


def infication(index_node_for_infication, infication_time, network, death_note):
    prob_of_death = 0.01820518643617639 + 0.2
    network.infected_nodes[index_node_for_infication] = 1
    network.susceptible_nodes[index_node_for_infication] = 0
    network.contagious_nodes[index_node_for_infication][0] = 1
    network.times_node_infication[index_node_for_infication] = infication_time
    if(np.random.uniform(0.0, 1.0) <= network.death_probabilities[i]):
        death_note[index_node_for_infication] = 1 #die with probability


def CTMC(network, death_note, treatment_time, critically_treatment_time, infection_rate = 0.01, time = 0):
    # do actions (threatment, death, etc)
    do_actions(time, network, death_note, treatment_time, critically_treatment_time)
    contagious_contacts = get_contagious_contacts(network, infection_rate)
    contagious_contacts_concentration = sum(contagious_contacts)
    if(contagious_contacts_concentration <= 0):
        # what can i do?
        pass
    index_node_for_infication = infication_roulette_wheel_choise(contagious_contacts,contagious_contacts_concentration)
    infication(index_node_for_infication, time, network, death_note)
    return get_time_to_infication(contagious_contacts_concentration)


def start_infication(number_of_infications, network, death_note):
    infication_time = 0
    for _ in range(number_of_infications):
        index_node_for_infication = np.random.randint(0, len(network.infected_nodes))
        infication(index_node_for_infication, infication_time, network, death_note)


def get_time_to_action(current_time, time_to_next_infication, time_step):
    new_time = min(current_time + time_to_next_infication, current_time + time_step - current_time % time_step)
    return round(new_time, 12), round(time_to_next_infication -  (new_time - current_time), 12)


def do_actions(time, network, death_note, treatment_time, critically_treatment_time):
    for i in range(len(network.infected_nodes)):
        if(network.infected_nodes[i] == 1 and death_note[i] == 1 and time >= network.times_node_infication[i]+critically_treatment_time):
            network.infected_nodes[i] = 0
            network.susceptible_nodes[i] = 0
            network.contagious_nodes[i] = 0
        if(network.infected_nodes[i] == 1 and death_note[i] == 0 and time >= network.times_node_infication[i]+treatment_time):
            network.infected_nodes[i] = 0
            network.susceptible_nodes[i] = 0
            network.contagious_nodes[i] = 0
    #print("infected_nodes: ", infected_nodes)
    #print(treatment_time)
            

def get_states_info(network, death_note):
    amount_of_infected = sum(network.infected_nodes)
    amount_of_susceptible = sum(network.susceptible_nodes)
    amount_of_contagious = sum(network.contagious_nodes.T[0])
    amount_of_critically_infected = sum([1 for have_to_die, infected in zip(death_note, network.infected_nodes) if(have_to_die == 1 and infected == 1)])
    amount_of_dead = sum([1 for have_to_die, infected in zip(death_note, network.infected_nodes) if(have_to_die == 1 and infected == 0)])
    return amount_of_infected, amount_of_susceptible, amount_of_contagious, amount_of_critically_infected, amount_of_dead


def simulation(graph_size, network_type, amount_of_contacts, infection_rate, number_of_infications, max_time, time_step, i, path):

    death_note = [0 for i in range(graph_size)]
    treatment_time = 10
    critically_treatment_time = 14

    all_time = time.time()
    graph = Network_model.create_network(graph_size, amount_of_contacts, network_type)
    network = Network(np.array(list(graph.get_adjacency())), Population.get_population())
    
    start_infication(number_of_infications, network, death_note)
    time_to_next_infication = get_time_to_infication(np.sum(get_contagious_contacts(network, infection_rate)))
    current_time = 0
    #times_to_infications = [time_to_next_infication]

    states_info = [["time", "amount_of_infected", "amount_of_susceptible", "amount_of_contagious", "amount_of_critically_infected", "amount_of_dead"]]
    
    #print("current_time: ", current_time)

    while(current_time < max_time):
        current_time, time_to_next_infication = get_time_to_action(current_time, time_to_next_infication, time_step)
        #print("current_time: ", current_time, "\t time to next infication: ", time_to_next_infication)
        if(time_to_next_infication == 0):
            time_to_next_infication = CTMC(network, death_note, treatment_time, critically_treatment_time, infection_rate, current_time)
            #times_to_infications.append(time_to_next_infication)
            #print("new time_to_next_infication: ", time_to_next_infication)
        if(current_time % time_step == 0):
            do_actions(current_time, network, death_note, treatment_time, critically_treatment_time)
            states_info.append([current_time] + list(get_states_info(network, death_note)))
            
    #print(times_to_infications)
    #plt = px.line(x=range(len(times_to_infications)),y=times_to_infications).show() 
    #plt.plot(range(len(times_to_infications)), times_to_infications)
    #plt.show()

    #pprint(states_info)

    print("all time: ", time.time() - all_time)
    with open(path + str(i) + '.txt', 'w') as file:
        for row in states_info:
            file.write(','.join([str(a) for a in row]) + '\n')
    

graph_size = (10 ** 3) * 1
network_type = 'Complete'
amount_of_contacts = 0
infection_rate = 0.002
number_of_infications = 1
max_time = 100
time_step = 1
amount_of_simulations = 100

folder_path = "/run/media/fedora_user/31614d99-e16f-45e1-8be5-e21723cf8199/projects/ManagingEpidemicOutbreak/Python-v.0.1/simulations/"
folder_name = "size: {}, network: {}, node_contacts: {}, infection_rate: {}/".format(
    graph_size, network_type, amount_of_contacts, infection_rate)

data_path = folder_path + folder_name
if(path.exists(data_path) == False):
    os.makedirs(data_path)
    print("Created!")

for i in range(0, amount_of_simulations):
    simulation(graph_size, network_type, amount_of_contacts, infection_rate, number_of_infications, max_time, time_step, i, data_path)
