#pragma once

#include "Tensor.h"
#include <vector>
#include <memory>

namespace yan {

class Layer {
public:
    virtual ~Layer() = default;
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(Tensor grad_output, float learning_rate) = 0;
    virtual void to(Device device) = 0;
};

class Linear : public Layer {
private:
    Tensor weights; // Shape: [in_features, out_features]
    Tensor biases;  // Shape: [1, out_features]
    Tensor last_input; // Stored for backward pass

public:
    Linear(size_t in_features, size_t out_features);
    Tensor forward(const Tensor& input) override;
    Tensor backward(Tensor grad_output, float learning_rate) override;
    void to(Device device) override;
};

class ReLU : public Layer {
private:
    Tensor last_input;
public:
    ReLU() : last_input({0}) {}
    Tensor forward(const Tensor& input) override;
    Tensor backward(Tensor grad_output, float learning_rate) override;
    void to(Device device) override;
};

class Sigmoid : public Layer {
private:
    Tensor last_output;
public:
    Sigmoid() : last_output({0}) {}
    Tensor forward(const Tensor& input) override;
    Tensor backward(Tensor grad_output, float learning_rate) override;
    void to(Device device) override;
};

class MLP {
private:
    std::vector<std::unique_ptr<Layer>> layers;

public:
    MLP();
    void add(std::unique_ptr<Layer> layer);
    Tensor forward(Tensor input);
    void backward(Tensor grad_output, float learning_rate);
    void to(Device device);
};

// Loss function
float mse_loss(const Tensor& predictions, const Tensor& targets, Tensor& grad_output);

} // namespace yan
