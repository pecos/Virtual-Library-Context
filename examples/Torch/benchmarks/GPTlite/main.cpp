/**
 * https://github.com/brunomaga/brunomaga.github.io/tree/master/assets/GPT-lite-cpp
 */
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <time.h>

#include "gptlite.h"

namespace F = torch::nn::functional;

// peform X warmup epochs before measuring performance on Y benchmark epochs 
const uint warmup_epochs = 30;
const uint benchmark_epochs = 30;
const torch::Device device = torch::kCPU;


template <typename ModelType>
void benchmark_train(ModelType & model, const torch::Tensor x, const torch::Tensor label, const std::string model_name) {
  torch::Tensor output, loss;
  
  model.train();
  torch::optim::Adam optimizer( model.parameters(),
    torch::optim::AdamOptions(2e-4).betas(std::make_tuple(0.5, 0.5)));

  for (int64_t epoch = 0; epoch < warmup_epochs + benchmark_epochs; ++epoch) {
    optimizer.zero_grad();
    output = model.forward(x);
    output = F::softmax(output, F::SoftmaxFuncOptions(1));
    loss = torch::cross_entropy_loss(output, label);
    loss.backward();
    optimizer.step();
  }
}


template <typename ModelType, typename InputType = torch::Tensor>
void benchmark_inference(ModelType & model, const InputType x, const std::string model_name, const uint epochs_multiplier=10) {
  { 
    torch::NoGradGuard no_grad; //no_grad scope, C++ equivalent to 'with torch.no_grad()' in Python

    for (int64_t epoch = 0; epoch < warmup_epochs*epochs_multiplier; ++epoch) 
      model.forward(x);

    for (int64_t epoch = 0; epoch < benchmark_epochs*epochs_multiplier; ++epoch)
      model.forward(x);
  }
}


int gptlite(int argc, const char* argv[], pthread_barrier_t* barrier) {
  {
    // GPTlite Model: (B, T, C) = (batch_size_deep, block_size, n_embed)
    const std::string model_name = "GPTlite";
    const int vocab_size = 65, batch_size=1; 
    const torch::ScalarType Long = torch::ScalarType::Long;
    torch::Tensor idx = torch::randint(0, vocab_size, {batch_size, block_size}, device).to(Long);
    torch::Tensor label = torch::randint(0, vocab_size, {batch_size, block_size}, device).to(Long);
    {
      GPTlite model = GPTlite(vocab_size, device);
      pthread_barrier_wait(barrier);
      auto start_time = std::chrono::system_clock::now();
      benchmark_train<GPTlite>(model, idx, label, model_name);
      // benchmark_inference<GPTlite>(model, idx, model_name);
      auto end_time = std::chrono::system_clock::now();
      std::cout << "gptlite runtime: " << ((std::chrono::duration<double>) (end_time - start_time)).count() << "s\n";
    }
  }
  return 0;
}