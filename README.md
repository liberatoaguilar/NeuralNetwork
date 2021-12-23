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

## Problems
* The network is not very fast.
* Optimizations such as training in batches could be added.
* For more complex training examples it seems to fall into saddle points easily.
    * This is somewhat mitigated by playing around with parameters and variable learning
      rates.

## Example Output
* Example output from main.cpp for an and gate
* The numbers represent the cost of the network
* Final prediction at the end
* ```
0.124594\n
0.100928\n
0.0904707\n
0.085413\n
0.0826221\n
0.0808024\n
0.0794089\n
0.0782104\n
0.0771079\n
0.0760587\n
0.0750441\n
0.074056\n
0.0730904\n
0.0721454\n
0.0712197\n
0.0703128\n
0.069424\n
0.0685527\n
0.0676986\n
0.0668612\n
0.0660402\n
0.065235\n
0.0644454\n
0.063671\n
0.0629115\n
0.0621665\n
0.0614357\n
0.0607188\n
0.0600155\n
0.0593318\n
0.0586604\n
0.0580021\n
0.0573565\n
0.0567233\n
0.0561023\n
0.0554932\n
0.0548956\n
0.0543095\n
0.0537344\n
0.0531702\n
0.0526165\n
0.0520733\n
0.0515402\n
0.051017\n
0.0505036\n
0.0499996\n
0.0495049\n
0.0490193\n
0.0485425\n
0.0480744\n
0.0476148\n
0.0471635\n
0.0467203\n
0.0462849\n
0.0458573\n
0.0454373\n
0.0450246\n
0.0446191\n
0.0442206\n
0.0438289\n
0.0434439\n
0.0430655\n
0.0426933\n
0.0423273\n
0.0419653\n
0.0416038\n
0.0412426\n
0.0408815\n
0.0405204\n
0.0401594\n
0.0397985\n
0.0394375\n
0.0390766\n
0.0387157\n
0.0383549\n
0.0379941\n
0.0376334\n
0.0372728\n
0.0369124\n
0.0365521\n
0.0361919\n
0.035832\n
0.0354723\n
0.0351129\n
0.0347537\n
0.0343949\n
0.0340364\n
0.0336783\n
0.0333205\n
0.0329633\n
0.0326065\n
0.0322503\n
0.0318946\n
0.0315396\n
0.0311851\n
0.0308314\n
0.0304784\n
0.0301262\n
0.0297749\n
0.0294244\n
0.0290748\n
0.0287262\n
0.0283787\n
0.0280322\n
0.0276869\n
0.0273427\n
0.0269998\n
0.0266582\n
0.0263179\n
0.025979\n
0.0256416\n
0.0253057\n
0.0249715\n
0.0246388\n
0.0243079\n
0.0239787\n
0.0236513\n
0.0233258\n
0.0230023\n
0.0226807\n
0.0223612\n
0.0220439\n
0.0217287\n
0.0214158\n
0.0211051\n
0.0207968\n
0.020491\n
0.0201876\n
0.0198868\n
0.0195886\n
0.019293\n
0.0190001\n
0.01871\n
0.0184227\n
0.0181383\n
0.0178568\n
0.0175782\n
0.0173027\n
0.0170302\n
0.0167609\n
0.0164947\n
0.0162318\n
0.015972\n
0.0157156\n
0.0154624\n
0.0152127\n
0.0149663\n
0.0147233\n
0.0144838\n
0.0142478\n
0.0140152\n
0.0137862\n
0.0135608\n
0.0133389\n
0.0131206\n
0.0129535\n
0.0127029\n
0.012481\n
0.0122776\n
0.0121666\n
0.0119394\n
0.0117334\n
0.011535\n
0.0113418\n
0.0111533\n
0.0109693\n
0.0107895\n
0.0106139\n
0.0104424\n
0.0102748\n
0.0101109\n
0.00995077\n
Input: 0 0 Prediction: 0 \n
Input: 1 0 Prediction: 0.0495656 \n
Input: 0 1 Prediction: 0.19744 \n
Input: 1 1 Prediction: 0.80787
```
