#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <vector>
#include <tuple>

// Configuration constants
int64_t SEQ_LEN = 64;
constexpr int64_t D_MODEL = 512;
int64_t NHEAD = 8;
constexpr int64_t NUM_LAYERS = 6;
int64_t VOCAB_SIZE = 267735;

class TextDataset : public torch::data::datasets::Dataset<TextDataset> {
public:
    explicit TextDataset(torch::Tensor data, int64_t seq_len) 
        : data_(data), seq_len_(seq_len) {}

    torch::data::Example<> get(size_t index) override {
        auto start = index;
        auto end = start + seq_len_ + 1;  // +1 for target
        return {data_.slice(0, start, end), torch::Tensor()};
    }

    torch::optional<size_t> size() const override {
        return data_.size(0) - seq_len_;
    }

private:
    torch::Tensor data_;
    int64_t seq_len_;
};

struct LanguageModel : torch::nn::Module {
    LanguageModel(float dropout) 
     : dropout(register_module("dropout", torch::nn::Dropout(dropout))),
       embedding(register_module("embed", torch::nn::Embedding(VOCAB_SIZE, D_MODEL))),
       transformer(register_module("trans", torch::nn::Transformer(
           torch::nn::TransformerOptions()
           .d_model(D_MODEL)
           .nhead(NHEAD)
           .num_encoder_layers(NUM_LAYERS)
           .num_decoder_layers(NUM_LAYERS)
           .dropout(dropout)))),
       classifier(register_module("cls", torch::nn::Linear(D_MODEL, VOCAB_SIZE)))
    {}

    torch::Tensor forward(torch::Tensor src) {
        auto emb = dropout->forward(embedding(src));
        emb = emb.transpose(0, 1);  // [SEQ_LEN, BATCH_SIZE, D_MODEL]

        auto mask = torch::ones({SEQ_LEN, SEQ_LEN}).triu(1).to(torch::kBool);

        auto output = transformer->forward(emb, emb, mask);

        output = output.transpose(0, 1);  // [BATCH_SIZE, SEQ_LEN, D_MODEL]
        output = classifier(output);      // [BATCH_SIZE, SEQ_LEN, VOCAB_SIZE]

        return torch::log_softmax(output, -1);
    }

    torch::nn::Dropout dropout{nullptr};
    torch::nn::Embedding embedding{nullptr};
    torch::nn::Transformer transformer{nullptr};
    torch::nn::Linear classifier{nullptr};
};

struct TrialResult {
    float learning_rate;
    float dropout;
    int batch_size;
    float final_loss;
    float validation_ppl;
};

TrialResult run_trial(
    float lr, 
    float dropout, 
    int batch_size,
    const TextDataset& train_set,
    const TextDataset& valid_set
) {
    // Create model and optimizer
    auto model = std::make_shared<LanguageModel>(dropout);
    torch::optim::Adam optimizer(model->parameters(), 
        torch::optim::AdamOptions(lr).betas({0.9, 0.98}));

    // Create data loaders
    auto train_loader = torch::data::make_data_loader(
        std::move(train_set),
        torch::data::DataLoaderOptions()
            .batch_size(batch_size)
            .drop_last(true)
    );

    auto valid_loader = torch::data::make_data_loader(
        std::move(valid_set),
        torch::data::DataLoaderOptions()
            .batch_size(batch_size)
            .drop_last(true)
    );

    // Training loop
    float final_loss = 0;
    for (int epoch = 0; epoch < 1; ++epoch) { // Short epochs for demo
        model->train();
        size_t count = 0;
        size_t total = train_set.size().value();
        for (auto& batch : *train_loader) {
            assert(batch.size() == batch_size);

            std::vector<torch::Tensor> input_list, target_list;
            for (const auto& b : batch) {
                input_list.emplace_back(b.data.slice(0, 0, SEQ_LEN));
                target_list.emplace_back(b.data.slice(0, 1, SEQ_LEN+1));
            }

            torch::Tensor inputs = torch::stack(input_list, 0); // [B, SEQ_LEN]
            torch::Tensor targets = torch::stack(target_list, 0).reshape(-1); // [B, 1]
            
            optimizer.zero_grad();
            auto outputs = model->forward(inputs).reshape({-1, VOCAB_SIZE});
            auto loss = torch::nll_loss(outputs, targets);
            loss.backward();
            torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
            optimizer.step();            
            final_loss = loss.item<float>();
            std::cout << "progress: " << (++count) * batch_size << "/" << total << std::endl;

            if (count * batch_size > 100) {
                break;
            }
        }
        std::cout << "Epoch: " << epoch << ", Loss: " << final_loss << std::endl;
    }

    // Validation
    model->eval();
    torch::NoGradGuard no_grad;
    float total_loss = 0;
    int count = 0;
    size_t total = valid_set.size().value();

    for (auto& batch : *train_loader) {
        assert(batch.size() == batch_size);

        std::vector<torch::Tensor> input_list, target_list;
        for (const auto& b : batch) {
            input_list.emplace_back(b.data.slice(0, 0, SEQ_LEN));
            target_list.emplace_back(b.data.slice(0, 1, SEQ_LEN+1));
        }

        torch::Tensor inputs = torch::stack(input_list, 0); // [B, SEQ_LEN]
        torch::Tensor targets = torch::stack(target_list, 0).reshape(-1); // [B, 1]
        
        auto outputs = model->forward(inputs).reshape({-1, VOCAB_SIZE});
        total_loss += torch::nll_loss(outputs, targets).item<float>();

        std::cout << "progress: " << (++count) * batch_size << "/" << total << std::endl;

        if (count * batch_size > 50) {
            break;
        }
    }

    return {lr, dropout, batch_size, final_loss, std::exp(total_loss / count)};
}

int transformer(int argc, const char* argv[], pthread_barrier_t* barrier) {
    NHEAD = argv[4] ? std::stoi(argv[4]) : NHEAD;
    SEQ_LEN = argv[5] ? std::stoi(argv[5]) : SEQ_LEN;
    // Hyperparameter search space
    std::vector<float> learning_rates = {argv[1] ? std::stof(argv[1]) : static_cast<float>(0.001)};
    std::vector<float> dropouts = {argv[2] ? std::stof(argv[2]) : static_cast<float>(0.1)};
    std::vector<int> batch_sizes = {argv[3] ? std::stoi(argv[3]) : 32};

    // Load the saved file
    auto container = torch::jit::load("wikitext2_processed.pt");

    auto train_data = container.attr("train").toTensor().to(torch::kLong);
    auto valid_data = container.attr("valid").toTensor().to(torch::kLong);
    const int64_t vocab_size = container.attr("vocab_size").toTensor().item<int64_t>();
    VOCAB_SIZE = vocab_size;

    TORCH_CHECK(train_data.dim() == 1, "Text data should be 1D tensor of token indices");

    // Create sliding window datasets
    auto train_set = TextDataset(train_data, SEQ_LEN);
    
    auto valid_set = TextDataset(valid_data, SEQ_LEN);

    std::vector<TrialResult> results;

    pthread_barrier_wait(barrier);
    auto start_time = std::chrono::system_clock::now();
    for (size_t lr_idx = 0; lr_idx < learning_rates.size(); ++lr_idx) {
        for (size_t do_idx = 0; do_idx < dropouts.size(); ++do_idx) {
            for (size_t bs_idx = 0; bs_idx < batch_sizes.size(); ++bs_idx) {         
                std::cout << "Running trial with lr=" << learning_rates[lr_idx]
                          << ", dropout=" << dropouts[do_idx]
                          << ", batch_size=" << batch_sizes[bs_idx] << "\n";  
                auto result = run_trial(
                    learning_rates[lr_idx],
                    dropouts[do_idx],
                    batch_sizes[bs_idx],
                    train_set,
                    valid_set
                );
                std::cout << "Trial completed. Result= " << result.final_loss << "\n";
                results.push_back(result);
            }
        }
    }
    auto end_time = std::chrono::system_clock::now();
    std::cout << "transformer runtime: " << ((std::chrono::duration<double>) (end_time - start_time)).count() << "s\n";

    // Find best configuration
    auto best = std::min_element(results.begin(), results.end(),
        [](const TrialResult& a, const TrialResult& b) {
            return a.validation_ppl < b.validation_ppl;
        });

    // Print results
    std::cout << "\nHyperparameter Tuning Results:\n";
    for (const auto& res : results) {
        std::cout << "lr=" << res.learning_rate 
                  << " dropout=" << res.dropout
                  << " bs=" << res.batch_size
                  << " | Loss: " << res.final_loss
                  << " | Val PPL: " << res.validation_ppl << "\n";
    }

    std::cout << "\nBest Configuration:\n"
              << "Learning Rate: " << best->learning_rate << "\n"
              << "Dropout: " << best->dropout << "\n"
              << "Batch Size: " << best->batch_size << "\n"
              << "Validation Perplexity: " << best->validation_ppl << std::endl;

    return 0;
}
