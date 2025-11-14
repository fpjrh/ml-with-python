import numpy as np # import Numpy library to generate 

# define a function to initlize the weights and biases of a neural network
def initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output):
    
    num_nodes_previous = num_inputs # number of nodes in the previous layer

    network = {}
    
    # loop through each layer and randomly initialize the weights and biases associated with each layer
    for layer in range(num_hidden_layers + 1):
        
        if layer == num_hidden_layers:
            layer_name = 'output' # name last layer in the network output
            num_nodes = num_nodes_output
        else:
            layer_name = 'layer_{}'.format(layer + 1) # otherwise give the layer a number
            num_nodes = num_nodes_hidden[layer] 
        
        # initialize weights and bias for each node
        network[layer_name] = {}
        for node in range(num_nodes):
            node_name = 'node_{}'.format(node+1)
            network[layer_name][node_name] = {
                'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
                'bias': np.around(np.random.uniform(size=1), decimals=2),
            }
    
        num_nodes_previous = num_nodes

    return network # return the network
#
def forward_propagate(network, inputs):
    
    layer_inputs = list(inputs) # start with the input layer as the input to the first hidden layer
    
    for layer in network:
        
        layer_data = network[layer]
        
        layer_outputs = [] 
        for layer_node in layer_data:
        
            node_data = layer_data[layer_node]
        
            # compute the weighted sum and the output of each node at the same time 
            node_output = node_activation(compute_weighted_sum(layer_inputs, node_data['weights'], node_data['bias']))
            layer_outputs.append(np.around(node_output[0], decimals=4))
            
        if layer != 'output':
            print('The outputs of the nodes in hidden layer number {} is {}'.format(layer.split('_')[1], layer_outputs))
    
        layer_inputs = layer_outputs # set the output of this layer to be the input to next layer

    network_predictions = layer_outputs
    return network_predictions
#
def compute_weighted_sum(inputs, weights, bias):
    return np.sum(inputs * weights) + bias
#
def node_activation(weighted_sum):
    return 1.0 / (1.0 + np.exp(-1 * weighted_sum))
#######################
#
weights = np.around(np.random.uniform(size=6), decimals=2) # initialize the weights
biases = np.around(np.random.uniform(size=3), decimals=2) # initialize the biases
# check the weights and biases
print(weights)
print(biases)
# Let's comput the output for a given input
x_1 = 0.5 # input 1
x_2 = 0.85 # input 2
# compute the weighted sum of inputs
z_11 = x_1 * weights[0] + x_2 * weights[1] + biases[0]
print('\nThe weighted sum of inputs at the first node in the hidden layer is {}'.format(z_11))
# compute the weighted sum of inputs at the second node
z_12 = x_1 * weights[2] + x_2 * weights[3] + biases[1]
print('The weighted sum of inputs at the second node in the hidden layer is {}'.format(z_12))
# assuming a sigmoid activation function, let's compute the activation of the first node
a_11 = 1.0 / (1.0 + np.exp(-z_11))
print('\nThe activation of the first node in the hidden layer is {}'.format(np.around(a_11, decimals=4)))
# let's compute the activation of the second node
a_12 = 1.0 / (1.0 + np.exp(-z_12))
print('The activation of the second node in the hidden layer is {}'.format(np.around(a_12, decimals=4)))
# compute the weighted sum of inputs at the output layer
z_2 = a_11 * weights[4] + a_12 * weights[5] + biases[2]
print('\nThe weighted sum of inputs at the output layer is {}'.format(np.around(z_2, decimals=4)))
# apply activation function at the output layer
a_2 = 1.0 / (1.0 + np.exp(-z_2))
print('The output of the network for x1 = 0.5 and x2 = 0.85 is {}'.format(np.around(a_2, decimals=4)))

# initalize a network
num_inputs = 5 # number of inputs
num_hidden_layers = 3 # number of hidden layers
num_nodes_hiddden = [3, 2, 3] # number of nodes in each hidden layer
num_nodes_output = 1 # number of nodes in the output layer

# initialize the weights and biases
small_network = initialize_network(num_inputs, num_hidden_layers, num_nodes_hiddden, num_nodes_output)
# print the network
for layer_name, layer in small_network.items(): # looping through each layer in the network
    print('\n{}\n'.format(layer_name), end='') # print layer name
    for node, weights in layer.items(): # looping through each node in the layer
        print('- {}: {} weights: '.format(node, weights['bias']), end='') # print node name and bias
        for weight in weights['weights']: # looping through each weight in the node
            print('{} '.format(weight), end='') # print weight
        print('')

#print("\n\nNetwork :", small_network) # print network
# generate  5 random inputs with 2 decimal places
inputs = np.around(np.random.uniform(size=5), decimals=2)
# compute the weighted sum for the first node in the first hidden layer
node_weights = small_network['layer_1']['node_1']['weights']
node_biases = small_network['layer_1']['node_1']['bias']
weighted_sum = compute_weighted_sum(inputs, node_weights, node_biases)
print('\nThe weighted sum at the first node in the hidden layer is {}'.format(np.around(weighted_sum, decimals=4)))

# compute the activation of the first node in the hidden layer
node_output = node_activation(weighted_sum)
print('The output of the first node in the hidden layer is {}'.format(np.around(node_output, decimals=4)))
print(' ')
# forward propagate the inputs through the network to get the predictions
predictions = forward_propagate(small_network, inputs)
print('\n\nThe predicted value by the network for the given input is {}'.format(np.around(predictions[0], decimals=4)))
#
# create a network with 5 inputs, 5 hidden layers and 1 output layer with 1 node.
# 5 inputs will be passed to the network with 3 random weights for each input
# Let's initialize the network
small_network = initialize_network(5, 5, [3, 3, 5, 4, 5], 1)
#
# print the network
for layer_name, layer in small_network.items(): # looping through each layer in the network
    print('\n{}\n'.format(layer_name), end='') # print layer name
    for node, weights in layer.items(): # looping through each node in the layer
        print('- {}: {} weights: '.format(node, weights['bias']), end='') # print node name and bias
        for weight in weights['weights']: # looping through each weight in the node
            print('{} '.format(weight), end='') # print weight
        print('')
# Let's generate 5 random inputs with 2 decimal places
inputs = np.around(np.random.uniform(size=5), decimals=2)
# compute the weighted sum for the first node in the first hidden layer
node_weights = small_network['layer_4']['node_3']['weights']
node_biases = small_network['layer_4']['node_3']['bias']
weighted_sum = compute_weighted_sum(inputs, node_weights, node_biases)
# compute the activation of the fourth node in the fourth hidden layer
node_output = node_activation(weighted_sum)
# forward propagate the inputs to get the predictions
predictions = forward_propagate(small_network, inputs)
# print the results
print('\n\nThe predicted value by the network for the given input is {}'.format(np.around(predictions[0], decimals=4)))

# initialize a single neuron network with 2 inputs and weights of 0.55 and 0.45 respectively
# and a bias of 0.15
single_neuron_network = initialize_network(2, 1, [1], 1)
node_weights = single_neuron_network['output']['node_1']['weights']
node_biases = single_neuron_network['output']['node_1']['bias']
print('\n\nThe weights of the single neuron network are {}'.format(node_weights))
