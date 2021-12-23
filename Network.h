#ifndef _NETWORK
#define _NETWORK

#include <iostream>
#include <vector>
#include <string>
#include <stdlib.h>
#include <time.h> 
#include <math.h>
#include <fstream>
#include "Neuron.h"
#include "Layer.h"

// Like a neuron but only used for dcdb and dcdw (gradient)
struct Simple_Node
{
    float dcdb{};
    std::vector<float> dcdw{};
    Simple_Node(int w)
    {
        for (int i = 0; i < w; ++i)
        {
            dcdw.push_back(0.0);
        }
    }
};

// Like a layer but only used for dcdb and dcdw
struct Simple_Layer
{
    std::vector<Simple_Node> nodes{};
    Simple_Layer(int w, int n)
    {
        for (int i = 0; i < n; ++i)
        {
            Simple_Node node(w);
            nodes.push_back(node);
        }
    }
};

// Copy of network to record dcdb and dcdw
struct Gradient
{
    std::vector<Simple_Layer> layers{};
    Gradient(int i, int o, int n, int l)
    {
        Simple_Layer first(i, n);
        layers.push_back(first);
        for (int i = 0; i < l-1; ++i)
        {
            Simple_Layer layer(n, n);
            layers.push_back(layer);
        }
        Simple_Layer out(n, o);
        layers.push_back(out);
    }
};

class Network
{
private:
    Layer* input;
    // All hidden layers
    std::vector<Layer*> hidden;
    Layer* output;

    int input_size = 0;
    int output_size = 0;
    int hidden_size = 0;
    int num_hidden = 0;
    float learning_rate = 0.1;

    // Different activation functions
    float sigmoid(float a) { return 1 / ( 1 + exp(-a) ); }
    float sigprime(float a) { return exp(-a)/((exp(-a)+1)*(exp(-a)+1)); }
    float relu(float a)  { return ( ( a > 0) ? a : 0 ); }
    float reluprime(float a) { return ( ( a > 0) ? 1 : 0 ); }
    float act(float a) { return relu(a); }
    float actprime(float a) { return reluprime(a); }
    //float act(float a) { return sigmoid(a); }
    //float actprime(float a) { return sigprime(a); }
    void update_layers(Layer* a, Layer* b);
    float calculate_cost(Layer* c);
    void calculate_error(Layer* a, Layer* b);
    void calculate_last_error(Layer* c);
    void calculate_all_errors(Layer* c);
    void calculate_gradient(Layer* c);
    void calculate_dcdw(Layer* a, Layer* b);
    void calculate_all_dcdw();
    void grad_descent(Layer* c);
    void update_weights_and_biases(Layer* a, Simple_Layer& s, int n);
    void record_gradient(Gradient& g);
    void get_weights_and_biases(Layer* a, Simple_Layer& l);
    void update(int n, Gradient& g);
    void backprop(Layer* c);

public:
    // Network from params or file 
    Network(int i, int o, int h, int nh);
    Network(std::string file);
    ~Network();

    Layer* predict(Layer* l);
    void train(std::string file, float count, int max_epoch, int freq);
    void test(std::string file);

    friend std::ostream& operator<<(std::ostream& os, Network& N);
};

std::ostream& operator<<(std::ostream& os, Network& N);

#endif

