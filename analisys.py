from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
from os import path


def get_data(directory, amount_of_files, extension):
    data = []
    for i in range(0,amount_of_files):
        data.append(genfromtxt(directory+ str(i) + '.' + extension, delimiter=',')[1:100])
    return np.array(data, dtype=int)


def get_mean(data, mean_type='average'):
    if(mean_type == 'average'):
        mean = np.mean(data, axis=0)
    if(mean_type == 'median'):
        mean = np.median(data, axis = 0)
    return mean


def plot(data, language, label):
    if(language == 'python' or language == 'Python'):
        index_1 = 0
        index_2 = 1
    if(language == 'r' or language == 'R'):
        index_1 = 1
        index_2 = 2
    plt.plot(data.T[index_1], data.T[index_2], label = label)


def get_path(parent_path, graph_size, network_type, amount_of_contacts, infection_rate):
    #Python only
    return parent_path + "size: {}, network: {}, node_contacts: {}, infection_rate: {}/".format(
    graph_size, network_type, amount_of_contacts, infection_rate)

mean_type = 'median'

np.set_printoptions(formatter={'float_kind':'{:f}'.format})

#python data
amount_of_files = 100
graph_size = 200 
network_type = 'Complete'
amount_of_contacts = 0
infection_rate = 0.002
parent_path = "/run/media/fedora_user/31614d99-e16f-45e1-8be5-e21723cf8199/projects/ManagingEpidemicOutbreak/Python-v.0.1/simulations/"
data_path = get_path(parent_path, graph_size, network_type, amount_of_contacts, infection_rate)
print(path.exists(data_path))
data = get_data(data_path, amount_of_files, 'txt')
mean = get_mean(data, mean_type)
plot(mean, 'Python', 'Python')


# data_R = get_data(parent_path + 'size 200, start_infected 1, infection_rates 0.002, time_step 1, time_max 100, network_type compleet_same, amount_of_edges 0, virus_types 1 death_rate_type same, prob_reconnection 0/data_', 36, 'csv')
# mean_R = get_mean(data_R, mean_type)
# plot(mean_R, 'R', 'R(newer)')

# data_R1 = get_data(parent_path + 'size 200, start_infected 1, infection_rates 0.002, time_step 1, time_max 100, network_type compleet, virus_types 1 death_rate_type same/data_', 32, 'csv')
# mean_R1 = get_mean(data_R1, mean_type)
# plot(mean_R1, 'R', 'R (older)')

plt.xlabel("day")
plt.ylabel("amount of people")

plt.title("Infected people (counting was provided ones per each \"day\")\n (average values)")

plt.legend()
plt.show()
