#include "model.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace strata::gigachat4;

namespace {
const std::string prefix = "model.layers.0.self_attn.";

void require(bool condition, const std::string& msg) {
    if (!condition) throw std::runtime_error(msg);
}

void close(const Vec& v, const Vec& expected, float tol, const std::string& msg) {
    require(v.size() == expected.size(), msg + " length");
    for (size_t i = 0; i < v.size(); ++i)
        require(std::abs(v[i] - expected[i]) <= tol,
                msg + "[" + std::to_string(i) + "] expected " + std::to_string(expected[i]) +
                ", got " + std::to_string(v[i]));
}

template<class F>
void rejects(F f, const char* msg) {
    try { f(); } catch (const std::exception&) { return; }
    throw std::runtime_error(std::string("accepted: ") + msg);
}

Config config() {
    return {{"hidden_size", 5}, {"linear_num_key_heads", 2}, {"linear_num_value_heads", 4},
            {"linear_key_head_dim", 3}, {"linear_value_head_dim", 2}, {"linear_conv_kernel_dim", 4},
            {"linear_attention_type", "GigaChat4GatedDeltaNet"}, {"hidden_act", "silu"},
            {"linear_use_short_conv", true}, {"linear_gating_type", "gated_rmsnorm_sigmoid"},
            {"linear_sigmoid_gate_scale", 2.0}, {"linear_attn_o_norm_eps", 1.0 / 256}, {"rms_norm_eps", 1e-6}};
}

float sample(size_t i, size_t j, size_t seed) {
    return (int((i * 13 + j * 7 + i * j * 3 + seed * (i + 2 * j + 1)) % 29) - 14) / 16.0f;
}

struct TestWeights : Weights {
    size_t hk = 2, hv = 4, dk = 3, dv = 2, hidden = 5, taps = 4;
    bool synthetic_projections = false;
    std::string truncate;
    size_t calls = 0;

    Vec linear(const std::string& name, const Vec& x, size_t out) override {
        size_t in, rows, seed;
        if (name == prefix + "in_proj_qkvz.weight") {
            in = hidden; rows = (2 * hk * dk + 2 * hv * dv); seed = 1;
        } else if (name == prefix + "in_proj_ba.weight") {
            in = hidden; rows = 2 * hv; seed = 2;
        } else if (name == prefix + "out_proj.weight") {
            in = hv * dv; rows = hidden; seed = 3;
        } else throw std::runtime_error("unexpected linear: " + name);
        require(x.size() == in && out == rows, name + " matrix shape");
        ++calls;
        Vec y(rows);
        for (size_t i = 0; i < rows; ++i) {
            if (synthetic_projections) {
                y[i] = x[i % in] * sample(i, 0, seed);  // Full widths, no large matrix.
            } else {
                for (size_t j = 0; j < in; ++j) y[i] += x[j] * sample(i, j, seed);
            }
        }
        if (name == prefix + truncate) y.pop_back();
        return y;
    }

    Vec read(const std::string& name, const std::vector<size_t>& shape) override {
        const size_t channels = 2 * hk * dk + hv * dv;
        Vec y;
        if (name == prefix + "conv1d.weight") {
            require(shape == std::vector<size_t>{channels, 1, taps}, name + " shape");
            for (size_t i = 0; i < channels; ++i)
                for (size_t t = 0; t < taps; ++t) y.push_back(sample(i, t, 4));
        } else if (name == prefix + "dt_bias" || name == prefix + "A_log") {
            require(shape == std::vector<size_t>{hv}, name + " shape");
            for (size_t i = 0; i < hv; ++i) y.push_back(sample(0, i, name == prefix + "dt_bias" ? 5 : 6));
        } else if (name == prefix + "norm.weight") {
            require(shape == std::vector<size_t>{dv}, name + " shape");
            for (size_t i = 0; i < dv; ++i) y.push_back(i % 2 == 0 ? 1.25f : -0.75f);
        } else throw std::runtime_error("unexpected read: " + name);
        if (name == prefix + truncate) y.pop_back();
        return y;
    }
    Vec row(const std::string&, size_t, size_t, size_t) override {
        throw std::runtime_error("GDN should not read matrix rows");
    }
};

Vec input(size_t step) {
    Vec x(5);
    for (size_t j = 0; j < x.size(); ++j)
        x[j] = sample(step, j, 7) * (step == 0 ? 1 / 16.0f : (step == 3 ? 0.0f : 1.0f));
    return x;
}

// FP32 goldens from the official chunk path; gdn_oracle.py also checks the
// official cached recurrent path against the same values.
const Vec expected_output = {
    2.664146200e-02f, 4.076194391e-02f, 5.422311835e-03f, 4.744667560e-03f, 7.318256889e-03f,
    3.416621387e-01f, 6.701202393e-01f, -5.533192754e-01f, -4.212599397e-01f, -7.112540305e-02f,
    -4.220306575e-01f, -1.005337089e-01f, -3.567630947e-01f, 2.175515890e-02f, 2.053309679e-01f,
    -5.520331860e-01f, 6.537564993e-01f, -1.355972052e+00f, 3.336609006e-01f, -1.074228436e-01f,
    3.763319850e-01f, -1.986962371e-02f, -4.419281185e-01f, -7.402001321e-02f, 2.063698024e-01f,
    -8.455248475e-01f, -2.889840305e-01f, -5.927478671e-01f, -1.734463722e-01f, -6.395588070e-02f,
};
const Vec expected_recurrent = {
    -1.277350634e-01f, 1.066317502e-02f, 6.887701154e-02f, 1.551110949e-02f, 7.450884581e-02f,
    -7.702203467e-03f, -6.469274871e-03f, 7.423356920e-02f, -3.413222730e-03f, 4.722310603e-02f,
    1.735736732e-04f, -6.120746024e-03f, -7.504444569e-02f, 4.572153091e-03f, 1.603404284e-01f,
    2.526296675e-02f, 5.507822335e-02f, 3.046764806e-02f, 4.978087544e-02f, -2.589574084e-03f,
    -5.840604007e-02f, 4.184037447e-02f, -3.772081435e-02f, 1.571808197e-02f,
};

void tiny_parity() {
    const Config c = config();
    TestWeights w;
    LayerState state, snapshot;
    Vec output;
    for (size_t t = 0; t < 6; ++t) {
        const Vec y = gdn_step(c, w, prefix, input(t), state);
        output.insert(output.end(), y.begin(), y.end());
        if (t == 2) snapshot = state;
        // Check every raw history tap, including left zero padding, oldest tap
        // eviction and the zero input token. Indices are projection rows in
        // convolution order: all Q, all K, all V. Z never enters the cache.
        const size_t rows[] = {0,1,2, 14,15,16, 3,4,5, 17,18,19, 6,7,8,9, 20,21,22,23};
        Vec raw;
        for (size_t row : rows) for (size_t tap = 0; tap < 4; ++tap) {
            float value = 0;
            if (t + tap >= 3) {
                const Vec prev = input(t + tap - 3);
                for (size_t j = 0; j < prev.size(); ++j) value += sample(row, j, 1) * prev[j];
            }
            raw.push_back(value);
        }
        close(state.conv, raw, 0, "raw conv cache");
    }
    close(output, expected_output, 2e-6f, "official outputs");
    close(state.recurrent, expected_recurrent, 1e-6f, "official final state");
    require(w.calls == 18, "three Weight::linear invocations per step");
    for (size_t t = 3; t < 6; ++t) {
        Vec want(output.begin() + t * 5, output.begin() + (t + 1) * 5);
        close(gdn_step(c, w, prefix, input(t), snapshot), want, 0, "restored cache");
    }
    close(snapshot.recurrent, state.recurrent, 0, "restored final state");
}

void validation() {
    const auto base = config();
    auto run = [&](Config c, LayerState& s, TestWeights& w) { gdn_step(c, w, prefix, input(0), s); };
    LayerState state;
    TestWeights w;
    const std::vector<std::pair<std::string, Config>> invalid = {
        {"linear_num_key_heads", 3}, {"linear_num_value_heads", -4}, {"linear_value_head_dim", 2.5},
        {"linear_key_head_dim", true}, {"linear_conv_kernel_dim", 0},
        {"hidden_act", "gelu"}, {"linear_gating_type", "gated_rmsnorm"}, {"linear_use_short_conv", false},
        {"linear_attention_type", "unknown"}, {"linear_attn_o_norm_eps", 0.0},
        {"linear_sigmoid_gate_scale", "2"}, {"linear_num_key_heads", std::numeric_limits<int64_t>::max()}};
    for (const auto& [field, bad] : invalid) {
        auto c = base;
        c[field] = bad;
        rejects([&] { run(c, state, w); }, field.c_str());
    }
    rejects([&] { gdn_step(base, w, prefix, Vec(6), state); }, "wrong hidden width");
    state.conv.resize(80);
    rejects([&] { run(base, state, w); }, "missing recurrent cache");
    state.recurrent.resize(23);
    rejects([&] { run(base, state, w); }, "wrong recurrent shape");
    state.conv.resize(79);
    state.recurrent.resize(24);
    rejects([&] { run(base, state, w); }, "wrong conv cache shape");
    require(w.calls == 0, "bad config/cache reached a projection");
    for (const char* name : {"in_proj_qkvz.weight", "in_proj_ba.weight", "conv1d.weight",
                             "dt_bias", "A_log", "norm.weight", "out_proj.weight"}) {
        state = {};
        w.truncate = name;
        rejects([&] { run(base, state, w); }, name);
    }
}

void checkpoint_geometry(Config c) {
    TestWeights w;
    w.hk = 32; w.hv = 64; w.dk = w.dv = 128; w.taps = 4; w.hidden = 7168;
    w.synthetic_projections = true;
    LayerState s;
    Vec x(w.hidden, 0.5f);
    auto y = gdn_step(c, w, prefix, x, s);
    require(s.conv.size() == 65536 && s.recurrent.size() == 1048576, "708B cache dimensions");
    const float* storage = s.recurrent.data();
    auto z = gdn_step(c, w, prefix, x, s);
    require(std::all_of(z.begin(), z.end(), [](float f) { return std::isfinite(f); }), "708B nonfinite");
    require(storage == s.recurrent.data() && y != z && w.calls == 6, "708B step/cache reuse");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        tiny_parity();
        validation();
        Config c = config();
        if (argc == 2) {
            std::ifstream input_config(argv[1]);
            input_config >> c;
        } else {
            c["hidden_size"] = 7168;
            c["linear_num_key_heads"] = 32; c["linear_num_value_heads"] = 64;
            c["linear_key_head_dim"] = c["linear_value_head_dim"] = 128;
            c["linear_attn_o_norm_eps"] = 1e-6;
        }
        checkpoint_geometry(c);
        std::cout << "GigaChat4 GDN: tiny oracle, cache/shape checks and 708B geometry passed\n";
    } catch (const std::exception& err) {
        std::cerr << err.what() << '\n';
        return 1;
    }
}
