#include <iostream>
#include <vector>
#include <string>
#include <stdlib.h>
#include <time.h> 
#include <cmath>
#include "Neuron.h"
#include "Layer.h"
#include "Network.h"

using std::cin, std::cout, std::endl;

// Example main file
int main()
{
    srand(time(NULL));

    // From trained network file
    // Network n("and_trained.txt");

    // New network
    // i o h nh
    Network n(2, 1, 2, 1);
    n.train("examples/t_and.txt", 0.01, 10000, 2);
    n.test("examples/and.txt");

    // Record network
    //std::ofstream f("and_trained.txt");
    //f << n;
    return 0;
}

