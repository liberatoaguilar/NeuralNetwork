# Neural Network

This is a neural network project created as way to understand the mathematics and theory
of neural networks. This project is implemented entirely from scratch using C++. It is
currently still a **work in progress**. This network is able to learn _and, or, and xor gates_ consistently. 


## Files

* `Network.cpp` and `Network.h` are the main drivers of the network. These files handle the creation of the layers and neurons, predicting outputs, and training using backpropagation.
* `Layer.cpp` and `Layer.h` store the array of neurons of any layer (input, output, hidden)
* `Node.cpp` and `Node.h` store the array of weights, a single bias, and a single
  activation. In the backpropagation stage partial derivatives of weights and biases are
  also stored in Node objects.
* `Main.cpp` shows an example of how a Network is created and used.
* The `examples` folder has examples of how a training file (no prefix/suffix), testing
  file (`t_` prefix), and already trained network file (`_trained`) suffix are used. The
  already trained files are weights and biases of a network that has already been
  trained. It is possible to initialize a Network with this file and verify that it
  works using `.test()`.

## Math
* The theory and math used to create this entire project comes from:
    * `https://www.3blue1brown.com/lessons/neural-networks`
    * YouTube Playlist `3blue1brown: Neural Networks`
    * `http://neuralnetworksanddeeplearning.com/index.html`
    * `https://www.machinecurve.com/index.php/2020/02/26/getting-out-of-loss-plateaus-by-adjusting-learning-rates/`


