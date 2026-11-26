#include "decoder.hpp"

#include <cmath>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>

namespace gc = strata::gigachat4;
using gc::Vec;

namespace {
void near(float a, float b) {
    if (!(std::abs(a - b) <= 3e-5f)) throw std::runtime_error("unexpected value " + std::to_string(a));
}
struct Dense : gc::Weights {
    std::map<std::string, Vec> tensors;
    Vec linear(const std::string& key, const Vec& x, size_t out) override {
        const auto& m = tensors.at(key);
        if (m.size() != x.size() * out) throw std::runtime_error("shape: " + key);
        Vec y(out);
        for (size_t i = 0; i < out; ++i)
            y[i] = std::inner_product(x.begin(), x.end(), m.begin() + i * x.size(), 0.f);
        return y;
    }
    Vec read(const std::string& key, const std::vector<size_t>&) override { return tensors.at(key); }
    Vec row(const std::string&, size_t, size_t, size_t) override { throw std::logic_error("unexpected embedding"); }
};
}

int main() try {
    // Unbiased probabilities .2, .6, .8. Correction promotes expert 0;
    // including that correction in the mixing weights gives a different answer.
    const auto r = gc::route_experts({std::log(.25f), std::log(1.5f), std::log(4.f)}, {.9f, 0, 0}, 2, 2.5f);
    if (r.size() != 2 || r[0].expert != 0 || r[1].expert != 2) throw std::runtime_error("wrong experts");
    near(r[0].weight, .5f);
    near(r[1].weight, 2.f);

    Dense w;
    w.tensors["gate_proj.weight"] = {18, -13};
    w.tensors["up_proj.weight"] = {-17, 20};
    w.tensors["down_proj.weight"] = {1, 3000};
    const float expected = -100.f / (1.f + std::exp(-10.f)) -
        3000.f * 130.f / (1.f + std::exp(13.f));
    near(gc::mlp(w, "", {1}, 2, 10)[0], expected);

    // The gating bottleneck uses the normalized, gamma-weighted input.
    w.tensors["weight"] = {2.f, .5f};
    w.tensors["gate_up_projection.weight"] = Vec(32);
    w.tensors["gate_up_projection.weight"][0] = 2.f;
    w.tensors["gate_down_projection.weight"] = Vec(32);
    w.tensors["gate_down_projection.weight"][16] = -3.f;
    const auto n = gc::gated_norm({{"rms_norm_eps", .5f}, {"layernorm_gating_weight", 2.f}}, w, "", {2,-1});
    const float a = 4.f / std::sqrt(3.f), b = -.5f / std::sqrt(3.f);
    near(n[0], a);
    near(n[1], b * 2.f / (1.f + std::exp(3.f * (2.f * a) / (1.f + std::exp(-2.f * a)))));
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
