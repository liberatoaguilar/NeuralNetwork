#include <iostream>
#include <vector>
#include <string>
#include <stdlib.h>
#include <time.h> 
#include "Neuron.h"
#include "Layer.h"
#include "Network.h"

using std::cin, std::cout, std::endl;

Neuron::Neuron(float a, float b, int nw)
{
    this->activation = a;
    this->bias = b;
    this->num_weights = nw;

    // Random weights
    for (int i = 0; i < this->num_weights; ++i)
    {
        float w = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        this->weight.push_back(w);
    }
}

// Setters

void Neuron::set_activation(float a)
{
    this->activation = a;
}

void Neuron::set_bias(float b)
{
    this->bias = b;
}

void Neuron::set_zval(float z)
{
    this->zval = z;
}

void Neuron::set_dcdb(float d)
{
    this->dcdb = d;
}

void Neuron::set_dcdw(std::vector<float>& d)
{
    this->dcdw = d;
}

void Neuron::set_weights(std::vector<float>& w)
{
    this->weight = w;
}


std::ostream& operator<<(std::ostream& os, Neuron& n)
{
    // Format is:
    // number weights bias activation

    // Output number of weights
    os << n.get_num_weights() << ' ';
    for (int i = 0; i < n.get_num_weights(); ++i)
    {
        // Output weights
        os << n.get_weights().at(i) << ' ';
    }
    // Output bias and activation
    os << n.get_bias() << ' ' << n.get_activation();
    return os;
}

std::istream& operator>>(std::istream& is, Neuron& n)
{
    // Neuron from file
    // Format is:
    // number weights bias activation
    float a;
    float b;
    std::vector<float> all_w;
    float w;
    float nw;
    is >> nw;
    n.num_weights = nw;
    for (int i = 0; i < nw; ++i)
    {
        is >> w;
        all_w.push_back(w);
    }
    is >> b;
    is >> a;
    n.set_activation(a);
    n.set_bias(b);
    n.set_weights(all_w);
    return is;
}
