#include "NN.h"
#include <iostream>
#include <iomanip>

using namespace yan;

int main() {
    std::cout << "--- Training XOR with Custom Deep Learning Framework ---\n";

    // Dataset: XOR Inputs and Targets
    Tensor inputs({4, 2}, {
        0, 0,
        0, 1,
        1, 0,
        1, 1
    });

    Tensor targets({4, 1}, {
        0,
        1,
        1,
        0
    });

    // Create MLP: 2 (input) -> 4 (hidden) -> 1 (output)
    MLP model;
    model.add(std::make_unique<Linear>(2, 4));
    model.add(std::make_unique<Sigmoid>());
    model.add(std::make_unique<Linear>(4, 1));
    model.add(std::make_unique<Sigmoid>());
    
    // Move to GPU
    inputs.to(Device::Metal);
    targets.to(Device::Metal);
    model.to(Device::Metal);

    // Training loop
    int epochs = 10000;
    float learning_rate = 0.5f;

    for (int epoch = 0; epoch <= epochs; ++epoch) {
        // Forward pass
        Tensor predictions = model.forward(inputs);

        // Compute loss
        Tensor grad_output({4, 1});
        grad_output.to(Device::Metal);
        
        // Note: mse_loss currently expects Tensors on CPU for indexing. 
        // We need to sync predictions and targets to CPU for loss calculation briefly.
        predictions.to(Device::CPU);
        targets.to(Device::CPU);
        grad_output.to(Device::CPU);
        
        float loss = mse_loss(predictions, targets, grad_output);
        
        // Move grad_output back to GPU for backward pass
        grad_output.to(Device::Metal);
        targets.to(Device::Metal); // put it back

        // Backward pass
        model.backward(grad_output, learning_rate);

        if (epoch % 1000 == 0) {
            std::cout << "Epoch " << epoch << " Loss: " << std::fixed << std::setprecision(5) << loss << "\n";
        }
    }

    std::cout << "\n--- Final Predictions (from GPU) ---\n";
    Tensor final_preds = model.forward(inputs);
    final_preds.to(Device::CPU);
    final_preds.print();

    std::cout << "\nTargets:\n";
    targets.print();

    return 0;
}
