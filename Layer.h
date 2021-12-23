#ifndef _LAYER
#define _LAYER

#include <iostream>
#include <vector>
#include <string>
#include <stdlib.h>
#include <time.h> 
#include "Neuron.h"

class Layer
{
private:
    // Holds neurons in a layer
    std::vector<Neuron*> neurons;
    int size = 0;

public:
    Layer(int s, int w);
    ~Layer();

    // Getters
    std::vector<Neuron*>& get_neurons() { return this->neurons; }
    float get_size() { return this->size; }
    // Pretty Print
    void p_print(std::ostream& os) const;

    // Output and Input operators
    friend std::istream& operator>>(std::istream& is, Layer& l);
    friend std::ostream& operator<<(std::ostream& os, Layer& l);
};

std::istream& operator>>(std::istream& is, Layer& l);
std::ostream& operator<<(std::ostream& os, Layer& l);

#endif
