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
0.124594
0.100928
0.0904707
0.085413
0.0826221
0.0808024
0.0794089
0.0782104
0.0771079
0.0760587
0.0750441
0.074056
0.0730904
0.0721454
0.0712197
0.0703128
0.069424
0.0685527
0.0676986
0.0668612
0.0660402
0.065235
0.0644454
0.063671
0.0629115
0.0621665
0.0614357
0.0607188
0.0600155
0.0593318
0.0586604
0.0580021
0.0573565
0.0567233
0.0561023
0.0554932
0.0548956
0.0543095
0.0537344
0.0531702
0.0526165
0.0520733
0.0515402
0.051017
0.0505036
0.0499996
0.0495049
0.0490193
0.0485425
0.0480744
0.0476148
0.0471635
0.0467203
0.0462849
0.0458573
0.0454373
0.0450246
0.0446191
0.0442206
0.0438289
0.0434439
0.0430655
0.0426933
0.0423273
0.0419653
0.0416038
0.0412426
0.0408815
0.0405204
0.0401594
0.0397985
0.0394375
0.0390766
0.0387157
0.0383549
0.0379941
0.0376334
0.0372728
0.0369124
0.0365521
0.0361919
0.035832
0.0354723
0.0351129
0.0347537
0.0343949
0.0340364
0.0336783
0.0333205
0.0329633
0.0326065
0.0322503
0.0318946
0.0315396
0.0311851
0.0308314
0.0304784
0.0301262
0.0297749
0.0294244
0.0290748
0.0287262
0.0283787
0.0280322
0.0276869
0.0273427
0.0269998
0.0266582
0.0263179
0.025979
0.0256416
0.0253057
0.0249715
0.0246388
0.0243079
0.0239787
0.0236513
0.0233258
0.0230023
0.0226807
0.0223612
0.0220439
0.0217287
0.0214158
0.0211051
0.0207968
0.020491
0.0201876
0.0198868
0.0195886
0.019293
0.0190001
0.01871
0.0184227
0.0181383
0.0178568
0.0175782
0.0173027
0.0170302
0.0167609
0.0164947
0.0162318
0.015972
0.0157156
0.0154624
0.0152127
0.0149663
0.0147233
0.0144838
0.0142478
0.0140152
0.0137862
0.0135608
0.0133389
0.0131206
0.0129535
0.0127029
0.012481
0.0122776
0.0121666
0.0119394
0.0117334
0.011535
0.0113418
0.0111533
0.0109693
0.0107895
0.0106139
0.0104424
0.0102748
0.0101109
0.00995077
Input: 0 0 Prediction: 0 

Input: 1 0 Prediction: 0.0495656 

Input: 0 1 Prediction: 0.19744 

Input: 1 1 Prediction: 0.80787 
```
