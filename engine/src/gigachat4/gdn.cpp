#include "model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace strata::gigachat4 {
namespace {

size_t dimension(const Config& c, const char* key) {
    const auto& n = c.at(key);
    if (!n.is_number_integer() || (!n.is_number_unsigned() && n.get<int64_t>() <= 0))
        throw std::invalid_argument(std::string("invalid GDN dimension: ") + key);
    const auto v = n.get<uint64_t>();
    if (v == 0 || v > Vec().max_size())
        throw std::invalid_argument(std::string("invalid GDN dimension: ") + key);
    return static_cast<size_t>(v);
}

// All dimensions are positive. Bound products *before* deriving offsets or
// asking the weight provider for a projection.
size_t product(size_t a, size_t b) {
    if (a > Vec().max_size() / b) throw std::length_error("GDN shape overflow");
    return a * b;
}

size_t sum(size_t a, size_t b) {
    if (a > Vec().max_size() - b) throw std::length_error("GDN shape overflow");
    return a + b;
}

float number(const Config& c, const char* key) {
    const auto& n = c.at(key);
    if (!n.is_number() || !std::isfinite(n.get<float>()))
        throw std::invalid_argument(std::string("invalid GDN parameter: ") + key);
    return n.get<float>();
}

struct Shape {
    const size_t hidden, hk, hv, dk, dv, taps;
    const size_t keys, values, channels, qkvz, ba, conv, recurrent;
    const float eps, gate_scale;

    explicit Shape(const Config& c)
        : hidden(dimension(c, "hidden_size")),
          hk(dimension(c, "linear_num_key_heads")), hv(dimension(c, "linear_num_value_heads")),
          dk(dimension(c, "linear_key_head_dim")), dv(dimension(c, "linear_value_head_dim")),
          taps(dimension(c, "linear_conv_kernel_dim")), keys(product(hk, dk)), values(product(hv, dv)),
          channels(sum(product(2, keys), values)), qkvz(sum(channels, values)), ba(product(2, hv)),
          conv(product(channels, taps)), recurrent(product(dk, values)),
          eps(number(c, c.contains("linear_attn_o_norm_eps") && !c.at("linear_attn_o_norm_eps").is_null()
                            ? "linear_attn_o_norm_eps" : "rms_norm_eps")),
          gate_scale(number(c, "linear_sigmoid_gate_scale")) {
        if (hv % hk != 0) throw std::invalid_argument("GDN value heads must be a multiple of key heads");
        if (eps <= 0) throw std::invalid_argument("GDN output norm epsilon must be positive");
        if (c.at("linear_attention_type") != "GigaChat4GatedDeltaNet" ||
            c.at("linear_gating_type") != "gated_rmsnorm_sigmoid" ||
            c.at("linear_use_short_conv") != true || c.at("hidden_act") != "silu")
            throw std::invalid_argument("unsupported GigaChat4 GDN configuration");
    }
};

Vec checked(Vec result, size_t expected, const std::string& name) {
    if (result.size() != expected)
        throw std::runtime_error("GDN weight provider returned wrong length: " + name);
    return result;
}

float sigmoid(float x) {
    const float t = std::exp(-std::abs(x));
    return x >= 0 ? 1.0f / (1.0f + t) : t / (1.0f + t);
}

float softplus(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }

float squares(const float* p, size_t n) {
    float acc = 0;
    for (size_t i = 0; i < n; ++i) acc += p[i] * p[i];
    return acc;
}

void l2_normalize(float* p, size_t n, float scale) {
    const float inv = 1.0f / std::sqrt(squares(p, n) + 1e-6f);
    for (size_t i = 0; i < n; ++i) p[i] = (p[i] * inv) * scale;
}

}  // namespace

Vec gdn_step(const Config& config, Weights& w, const std::string& prefix, const Vec& x, LayerState& state) {
    const Shape s(config);
    if (x.size() != s.hidden) throw std::invalid_argument("wrong GDN input width");
    const bool fresh = state.conv.empty() && state.recurrent.empty();
    if (!fresh && (state.conv.size() != s.conv || state.recurrent.size() != s.recurrent))
        throw std::invalid_argument("wrong GDN cache shape");

    auto linear = [&](const char* suffix, const Vec& input, size_t outputs) {
        const auto name = prefix + suffix;
        return checked(w.linear(name, input, outputs), outputs, name);
    };
    auto read = [&](const char* suffix, const std::vector<size_t>& dims, size_t n) {
        const auto name = prefix + suffix;
        return checked(w.read(name, dims), n, name);
    };
    const Vec qkvz = linear("in_proj_qkvz.weight", x, s.qkvz);
    const Vec ba = linear("in_proj_ba.weight", x, s.ba);
    const Vec kernel = read("conv1d.weight", {s.channels, 1, s.taps}, s.conv);
    const Vec dt = read("dt_bias", {s.hv}, s.hv);
    const Vec a_log = read("A_log", {s.hv}, s.hv);
    const Vec norm = read("norm.weight", {s.dv}, s.dv);

    Vec qkv(s.channels), out(s.values), delta(s.dv);
    if (fresh) {
        state.conv.resize(s.conv);
        state.recurrent.resize(s.recurrent);
    }
    // HF's saved history is [channel,tap], raw and oldest first. Projection
    // rows are grouped by key head; convolution channels are [all Q, all K, all V].
    auto convolve = [&](size_t first_channel, const float* src, size_t n) {
        for (size_t t = 0; t < n; ++t) {
            const size_t channel = first_channel + t;
            float* history = state.conv.data() + channel * s.taps;
            const float* weight = kernel.data() + channel * s.taps;
            float acc = 0;
            for (size_t i = 0; i + 1 < s.taps; ++i) {
                history[i] = history[i + 1];
                acc += history[i] * weight[i];
            }
            history[s.taps - 1] = src[t];
            acc += src[t] * weight[s.taps - 1];
            qkv[channel] = acc * sigmoid(acc);
        }
    };
    const size_t repeat = s.hv / s.hk;
    const size_t group_v = repeat * s.dv, group_size = s.qkvz / s.hk;
    const float q_scale = 1.0f / std::sqrt(static_cast<float>(s.dk));
    for (size_t group = 0; group < s.hk; ++group) {
        const float* packed = qkvz.data() + group * group_size;
        const size_t q_offset = group * s.dk, k_offset = s.keys + q_offset;
        const size_t v_offset = 2 * s.keys + group * group_v;
        convolve(q_offset, packed, s.dk);
        convolve(k_offset, packed + s.dk, s.dk);
        convolve(v_offset, packed + 2 * s.dk, group_v);
        float* q = qkv.data() + q_offset;
        float* k = qkv.data() + k_offset;
        l2_normalize(q, s.dk, q_scale);
        l2_normalize(k, s.dk, 1.0f);

        for (size_t r = 0; r < repeat; ++r) {
            const size_t h = group * repeat + r;
            const float* v = qkv.data() + v_offset + r * s.dv;
            const float* z = packed + 2 * s.dk + group_v + r * s.dv;
            const size_t b = group * (2 * repeat) + r;
            const float beta = sigmoid(ba[b]);
            const float decay = std::exp(-std::exp(a_log[h]) * softplus(ba[b + repeat] + dt[h]));
            float* matrix = state.recurrent.data() + h * s.dk * s.dv;
            float* y = out.data() + h * s.dv;

            // First contract the *decayed old* state with k. No row may
            // incorporate its correction until delta is known for all columns.
            std::fill(delta.begin(), delta.end(), 0.0f);
            for (size_t i = 0; i < s.dk; ++i) {
                float* row = matrix + i * s.dv;
                for (size_t j = 0; j < s.dv; ++j) {
                    row[j] *= decay;
                    delta[j] += k[i] * row[j];
                }
            }
            for (size_t j = 0; j < s.dv; ++j) delta[j] = (v[j] - delta[j]) * beta;
            // Then update and read the new state, streaming contiguous rows.
            for (size_t i = 0; i < s.dk; ++i) {
                float* row = matrix + i * s.dv;
                for (size_t j = 0; j < s.dv; ++j) {
                    row[j] += k[i] * delta[j];
                    y[j] += q[i] * row[j];
                }
            }
            const float inv = 1.0f / std::sqrt(squares(y, s.dv) / static_cast<float>(s.dv) + s.eps);
            for (size_t j = 0; j < s.dv; ++j)
                y[j] = ((y[j] * inv) * norm[j]) * sigmoid(z[j]) * s.gate_scale;
        }
    }
    return linear("out_proj.weight", out, s.hidden);
}

}  // namespace strata::gigachat4
