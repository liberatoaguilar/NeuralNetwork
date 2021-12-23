#include <iostream>
#include <vector>
#include <string>
#include <stdlib.h>
#include <time.h> 
#include <fstream>
#include <math.h>
#include "Neuron.h"
#include "Layer.h"
#include "Network.h"

using std::cin, std::cout, std::endl;

Network::Network(int i, int o, int h, int nh)
{
    this->input_size = i;
    this->output_size = o;
    this->hidden_size = h;
    this->num_hidden = nh;

    // First and last layer could have different sizes than hidden layers
    this->input = new Layer(this->input_size, 0);
    this->output = new Layer(this->output_size, this->hidden_size);

    // First hidden layer adapted to input weights
    Layer* l = new Layer(this->hidden_size, this->input_size);
    this->hidden.push_back(l);

    // All other hidden layers
    for (int i = 1; i < this->num_hidden; ++i)
    {
        Layer* l = new Layer(this->hidden_size, this->hidden_size);
        this->hidden.push_back(l);
    }
}

Network::Network(std::string file)
{
    // Fill network from file with format:
    // input output hidden_size num_hidden
    // Layer size
    // Neurons
    // Etc.
    std::ifstream f(file);
    int input_size;
    int output_size;
    int hidden_size;
    int num_hidden;
    f >> input_size >> output_size >> hidden_size >> num_hidden;
    this->input_size = input_size;
    this->output_size = output_size;
    this->hidden_size = hidden_size;
    this->num_hidden = num_hidden;
    for (int i = 0; i < this->num_hidden; ++i)
    {
        Layer* l = new Layer(0, 0);
        f >> *l;
        this->hidden.push_back(l);
    }
    Layer* o = new Layer(0, 0);
    f >> *o;
    this->output = o;
    this->input = new Layer(this->input_size, 0);
}

Network::~Network()
{
    // Delete layer pointers
    delete this->input;
    delete this->output;
    for (int i = 0; i < this->num_hidden; ++i)
    {
        delete this->hidden.at(i);
    }
}

void Network::update_layers(Layer* a, Layer* b)
{
    // Basically a matrix multiplication
    // Updates the neurons of layer b
    for (int i = 0; i < b->get_neurons().size(); ++i)
    {
        float new_act = 0.0;
        for (int j = 0; j < b->get_neurons().at(i)->get_num_weights(); ++j)
        {
            // Keep adding activations * weight
            new_act += (a->get_neurons().at(j)->get_activation()
                    * b->get_neurons().at(i)->get_weights().at(j));
        }
        // Update bias
        new_act += b->get_neurons().at(i)->get_bias();
        // Apply activation function and set activation
        b->get_neurons().at(i)->set_activation(this->act(new_act));
        // Z_val is without activation function (need for backpropagation)
        b->get_neurons().at(i)->set_zval(new_act);
    }
}

Layer* Network::predict(Layer* l)
{
    // Go through and update all weights and biases with given layer parameter
    this->update_layers(l, this->hidden.at(0));
    for (int i = 0; i < this->hidden.size()-1; ++i)
    {
        this->update_layers(this->hidden.at(i), this->hidden.at(i+1));
    }
    this->update_layers(this->hidden.at(this->hidden.size()-1), this->output);
    return this->output;
}

float Network::calculate_cost(Layer* c) { 
    // Calculates the cost function which is ((y-a)^2)/2
    // Where y is the expected output from layer c (correct layer)
    // And a is the activation of the output layer
    // This calculates the average of all the output layer

    float sum = 0.0;
    for (int i = 0; i < this->output->get_neurons().size(); ++i)
    {
        float y = c->get_neurons().at(i)->get_activation();
        float a = this->output->get_neurons().at(i)->get_activation();
        float c = (y-a)*(y-a);
        sum += c;
    }
    return sum/2;
}

void Network::calculate_last_error(Layer* c)
{
    // Calculates the error (partial derivative of cost with respect to bias) of last
    // layer
    // The equation is: (a-y)*actprime(z)
    // Where a is the activation, y is the expected output, z is the z value stored
    // when updating the neurons, and actprime is the derivative of the activtion function
    for (int i = 0; i < this->output->get_neurons().size(); ++i)
    {
        float y = c->get_neurons().at(i)->get_activation();
        float a = this->output->get_neurons().at(i)->get_activation();
        float z = this->output->get_neurons().at(i)->get_zval();
        float d = (a-y)*this->actprime(z);
        // Store dcdb of each node
        this->output->get_neurons().at(i)->set_dcdb(d);
    }
}

void Network::calculate_error(Layer* a, Layer*b)
{
    // Calculates the error (partial derivative of cost with respect to bias) of layer b
    // Assuimng that layer a's dcdb has been calculated
    // This is a backpropagation step
    // Layer a is after layer b
    std::vector<float> all_wd;
    for (int i = 0; i < a->get_neurons().at(0)->get_weights().size(); ++i)
    {
        // wd is the sum of all neurons' weights * dcdb
        float wd = 0.0;  
        for (int j = 0; j < a->get_neurons().size(); ++j)
        {
            float w = a->get_neurons().at(j)->get_weights().at(i);
            float d = a->get_neurons().at(j)->get_dcdb();
            wd += w*d;
        }
        // store in vector
        all_wd.push_back(wd);
    }
    // Calculate dcdb for layer b
    for (int i = 0; i < b->get_neurons().size(); ++i)
    {
        float z = b->get_neurons().at(i)->get_zval();
        // Actprime is derivative of activation function
        // w*d*actprime(z) similar to calculate_last_error()
        float d = all_wd.at(i)*this->actprime(z);
        b->get_neurons().at(i)->set_dcdb(d);
    }
}

void Network::calculate_all_errors(Layer *c)
{
    // First calculate last error
    this->calculate_last_error(c);
    // Loop backwards through each layer, updating dcdb
    calculate_error(this->output, this->hidden.at(this->hidden.size()-1));
    for (int i = this->hidden.size()-1; i > 0; --i)
    {
        calculate_error(this->hidden.at(i), this->hidden.at(i-1));
    }
}

void Network::calculate_dcdw(Layer* a, Layer* b)
{
    // Dcdw is the partial derivative of cost with respect to each weight
    for (int i = 0; i < b->get_neurons().size(); ++i)
    {
        std::vector<float> dcdw;
        for (int j = 0; j < b->get_neurons().at(i)->get_num_weights(); ++j)
        {
            float act = a->get_neurons().at(j)->get_activation();
            float d = b->get_neurons().at(i)->get_dcdb();
            // dcdw is activation * dcdb of the node
            float ad = act*d;
            dcdw.push_back(ad);
        }
        // Each neuron stores a vector of dcdw's
        // Unlike dcdb which is just a float
        b->get_neurons().at(i)->set_dcdw(dcdw);
    }
}

void Network::calculate_all_dcdw()
{
    // Calculate dcdw looping in a forward direction
    // Unlike dcdb which goes backwards
    this->calculate_dcdw(this->input, this->hidden.at(0));
    for (int i = 0; i < this->hidden.size()-1; ++i)
    {
        this->calculate_dcdw(this->hidden.at(i), this->hidden.at(i+1));
    }
    this->calculate_dcdw(this->hidden.at(this->hidden.size()-1), this->output);
}

void Network::backprop(Layer* c)
{
    // Calculate dcdb first, then dcdw for the entire network
    this->calculate_all_errors(c);
    this->calculate_all_dcdw();
}

void Network::grad_descent(Layer* c)
{
    this->backprop(c);
}

void Network::update_weights_and_biases(Layer* a, Simple_Layer& s, int n)
{
    // Update layer a with recorded gradient for this epoch
    for (int j = 0; j < a->get_neurons().size(); ++j)
    {
        float b = a->get_neurons().at(j)->get_bias();
        float dcdb = s.nodes.at(j).dcdb;
        // Update bias with learning rate and average dcdb in this layer
        a->get_neurons().at(j)->set_bias(b - this->learning_rate*(dcdb/n));
        for (int k = 0; k < a->get_neurons().at(j)->get_dcdw().size(); ++k)
        {
            float w = a->get_neurons().at(j)->get_weights().at(k);
            float dcdw = s.nodes.at(j).dcdw.at(k);
            // Update weights with learning rate and average dcdw for each node in layer
            a->get_neurons().at(j)->get_weights().at(k) = w - this->learning_rate*(dcdw/n);
        }
    }
}

void Network::update(int n, Gradient& avg)
{
    // Loop through recorded gradient for this epoch
    // n represents the number of training expames
    // It will be used to calculate average gradient
    // Because the gradient avg keeps a running sum
    for (int i = 0; i < this->hidden.size(); ++i)
    {
        this->update_weights_and_biases(this->hidden.at(i), avg.layers.at(i), n);
    }
    this->update_weights_and_biases(this->output, avg.layers.at(avg.layers.size()-1), n);
}

void Network::get_weights_and_biases(Layer* a, Simple_Layer& l)
{
    for (int j = 0; j < a->get_neurons().size(); ++j)
    {
        float dcdb = a->get_neurons().at(j)->get_dcdb();
        // Keep running sum of dcdb for each epoch of training
        l.nodes.at(j).dcdb += dcdb;
        for (int k = 0; k < a->get_neurons().at(j)->get_dcdw().size(); ++k)
        {
            float dcdw = a->get_neurons().at(j)->get_dcdw().at(k);
            // Keep running sum of dcdw for each epoch of training
            l.nodes.at(j).dcdw.at(k) += dcdw;
        }
    }
}

void Network::record_gradient(Gradient& g)
{
    // Go through entire network and gradient (copy of network with only dcdb and dcdw)
    for (int i = 0; i < this->hidden.size(); ++i)
    {
        this->get_weights_and_biases(this->hidden.at(i), g.layers.at(i));
    }
    this->get_weights_and_biases(this->output, g.layers.at(g.layers.size()-1));
}

void Network::train(std::string file, float cost, int max_epoch, int freq)
{
    // Really big number
    float avg_cost = 100000.0;
    // Epoch counter
    int epoch = 0;
    // Used for variable learning rate (to escape saddle points)
    float step = (freq*3.14)/max_epoch;
    float x = 0.0;

    // Either max epoch or cost achieved
    while (avg_cost > cost && epoch < max_epoch)
    {
        std::ifstream f(file);
        int training_size;
        f >> training_size;

        avg_cost = 0.0;
        // Gradient records the network's dcdb and dcdw at each epoch
        // These are used to update weights and biases
        // In theory this is the negative gradient which moves toward local min of cost
        Gradient g(this->input_size, this->output_size, this->hidden_size, this->num_hidden);

        // For each item in the training data
        for (int i = 0; i < training_size; ++i)
        {
            // Training data has input layer l and expected output layer c
            Layer* l = new Layer(0, 0);
            f >> *l;
            Layer* c = new Layer(0, 0);
            f >> *c;

            // Predict output with current state of network
            this->predict(l);
            // Keep running average cost
            avg_cost += this->calculate_cost(c);
            // Calculate dcdb and dcdw (backpropagation)
            this->grad_descent(c);
            // Record gradient for each item (keeps a running sum of dcdb, dcdw)
            this->record_gradient(g);

            delete l;
            delete c;
        }
        // Recalculate average cost
        avg_cost = avg_cost / training_size;
        // See how cost decreases
        cout << avg_cost << endl;
        // Update the weights and biases based on the final gradient sum (averaged)
        this->update(training_size, g);
        ++epoch;
        // Recalculate learning rate
        this->learning_rate = 0.4*sin(10*x-(3.14/2))+0.5;
        x += step;
    }
}

void Network::test(std::string file)
{
    // Get input from a file
    std::ifstream f(file);
    int test_size;
    f >> test_size;
    for (int i = 0; i < test_size; ++i)
    {
        // Predict for each input
        // Output nicely
        Layer* l = new Layer(0, 0);
        f >> *l;
        cout << "Input: ";
        l->p_print(cout);
        cout << "Prediction: ";
        this->predict(l)->p_print(cout);
        cout << endl << endl;
        delete l;
    }
}

std::ostream& operator<<(std::ostream& os, Network& n)
{
    // Used to output entire network's weights and biases to a text file
    os << n.input_size << ' ' << n.output_size << ' ' << n.hidden_size << ' ' << n.num_hidden << endl;
    for (int i = 0; i < n.hidden.size(); ++i)
    {
        os << *n.hidden.at(i);
    }
    os << *n.output;
    return os;
}
