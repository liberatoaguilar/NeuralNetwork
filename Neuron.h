#ifndef _NEURON
#define _NEURON

#include <iostream>
#include <vector>
#include <string>
#include <stdlib.h>
#include <time.h> 

class Neuron
{
private:
    float activation;
    float zval;
    float bias;
    float dcdb = -1;
    std::vector<float> weight{};
    std::vector<float> dcdw{};
    int num_weights = 0;

public:
    Neuron(float a) : activation(a) {};
    Neuron(float a, float b, int nw);

    float get_activation() const { return this->activation; }
    float get_zval() const { return this->zval; }
    float get_bias() const { return this->bias; }
    int get_num_weights() const { return this->num_weights; }
    float get_dcdb() const { return this->dcdb; }
    std::vector<float>& get_weights() { return this->weight; }
    std::vector<float>& get_dcdw() { return this->dcdw; }

    void set_activation(float a);
    void set_bias(float b);
    void set_zval(float z);
    void set_dcdb(float d);
    void set_dcdw(std::vector<float>& d);
    void set_weights(std::vector<float>& w);

    friend std::istream& operator>>(std::istream& is, Neuron& n);
    friend std::ostream& operator<<(std::ostream& os, Neuron& n);
};

std::istream& operator>>(std::istream& is, Neuron& n);
std::ostream& operator<<(std::ostream& os, Neuron& n);

#endif

