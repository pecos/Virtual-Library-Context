#include <torch/torch.h>
#include <iostream>

int main() {
  std::cout << "PyTorch version: "
    << TORCH_VERSION_MAJOR << std::endl;
}