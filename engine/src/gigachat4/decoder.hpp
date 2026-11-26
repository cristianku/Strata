#pragma once

#include "model.hpp"

namespace strata::gigachat4 {

struct Route {
    size_t expert;
    float weight;
};

std::vector<Route> route_experts(const Vec& logits, const Vec& correction, size_t k, float scale);
Vec gated_norm(const Config&, Weights&, const std::string& prefix, const Vec& input);
Vec mlp(Weights&, const std::string& prefix, const Vec& input, size_t intermediate, float clip);

class Decoder {
public:
    Decoder(Config config, Weights& weights);
    // One unpadded sequence. Pass a trace only with small fixtures: it records
    // intermediate activations and full-vocabulary logits for parity checks.
    Vec step(size_t token, Config* trace = nullptr, Config* routes = nullptr);

private:
    Config config_;
    Weights& weights_;
    std::vector<LayerState> layers_;
    std::vector<bool> sparse_;
    size_t position_ = 0;
    bool failed_ = false;
};

}  // namespace strata::gigachat4
