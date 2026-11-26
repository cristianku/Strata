#pragma once

#include "nlohmann/json.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace strata::gigachat4 {

using Vec = std::vector<float>;
using Config = nlohmann::json;

// File tensors are row major. linear() computes [out,in] @ x and must
// reject an unexpected shape. read() is for small weights, row() for embeddings.
class Weights {
public:
    virtual ~Weights() = default;
    virtual Vec linear(const std::string& name, const Vec& x, size_t out) = 0;
    virtual Vec read(const std::string& name, const std::vector<size_t>& shape) = 0;
    virtual Vec row(const std::string& name, size_t rows, size_t cols, size_t index) = 0;
};

// One state per sequence per layer. Prefill is repeated causal step execution.
struct LayerState {
    Vec conv, recurrent;
    Vec keys, values, index_keys;
};

// prefix is the full tensor prefix ending in ".self_attn.".
Vec gdn_step(const Config&, Weights&, const std::string& prefix, const Vec&, LayerState&);
Vec dsa_step(const Config&, Weights&, const std::string& prefix, const Vec&, LayerState&, size_t position);

}  // namespace strata::gigachat4
