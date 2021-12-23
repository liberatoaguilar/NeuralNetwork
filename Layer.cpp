#include <iostream>
#include <vector>
#include <string>
#include <stdlib.h>
#include <time.h> 
#include <math.h>
#include "Neuron.h"
#include "Layer.h"
#include "Network.h"

using std::cin, std::cout, std::endl;

Layer::Layer(int s, int w)
{
    this->size = s;

    // Initialze all neurons with random activations and biases
    for (int i = 0; i < this->size; ++i)
    {
        float act = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        float bia = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        Neuron* n = new Neuron(act, bia, w);
        this->neurons.push_back(n);
    }
}

Layer::~Layer()
{
    // Delete dynamic neurons
    for (int i = 0; i < this->size; ++i)
    {
        delete this->neurons.at(i);
    }
}

std::ostream& operator<<(std::ostream& os, Layer& l)
{
    // Format for layer file is:
    // size of layer (number of neurons)
    // Each neuron on a new line 
    os << l.get_size() << '\n';
    for (int i = 0; i < l.size; ++i)
    {
        os << *(l.neurons.at(i)) << endl;
    }
    return os;
}

void Layer::p_print(std::ostream& os) const
{
    // Only print the activation of the neurons
    for (int i = 0; i < this->size; ++i)
    {
        os << (this->neurons.at(i)->get_activation()) << ' ';
    }
}


std::istream& operator>>(std::istream& is, Layer& l)
{
    // Fill layer from file with format:
    // size of layer (number of neurons)
    // Each neuron on a new line 
    float size;
    is >> size;
    std::vector<Neuron*> neurons;
    for (int i = 0; i < size; ++i)
    {
        Neuron* n = new Neuron(0, 0, 0);
        is >> *n;
        neurons.push_back(n);
    }
    l.neurons = neurons;
    l.size = size;

    return is;
}
