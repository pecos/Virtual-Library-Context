/**
 * https://github.com/brunomaga/brunomaga.github.io/tree/master/assets/GPT-lite-cpp
 */
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <time.h>

#include "benchmark.h"

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


int dnn(int argc, const char* argv[], pthread_barrier_t* barrier) {
  {
    // Deep DNN model (W=256, L=2048)
    const std::string model_name = "Deep DNN";
    const int W=256, L=20, batch_size=2048;
    const int in_size=W, out_size=W;
    torch::Tensor x = torch::randn({batch_size, in_size}, device);
    torch::Tensor label = torch::randn({batch_size, out_size}, device);
    {
      BenchmarkModel model = BenchmarkModel(W, L, in_size, out_size, device);
      pthread_barrier_wait(barrier);
      auto start_time = std::chrono::system_clock::now();
      benchmark_train<BenchmarkModel>(model, x, label, model_name);
      // benchmark_inference<BenchmarkModel>(model, x, model_name);
      auto end_time = std::chrono::system_clock::now();
      std::cout << "dnn runtime: " << ((std::chrono::duration<double>) (end_time - start_time)).count() << "s\n";
    }
  }
  return 0;
}