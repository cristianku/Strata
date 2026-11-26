#include "decoder.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace strata::gigachat4 {
namespace {
void require(bool valid, const std::string& why) {
    if (!valid) throw std::invalid_argument("gigachat4: " + why);
}
size_t dimension(const Config& cfg, const char* key) {
    const auto& v = cfg.at(key);
    require(v.is_number_integer() && v.get<int64_t>() > 0, std::string("invalid ") + key);
    return v.get<size_t>();
}
float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}
void add(Vec& a, const Vec& b, float weight = 1.f) {
    require(a.size() == b.size(), "residual width mismatch");
    for (size_t i = 0; i < a.size(); ++i) a[i] += weight * b[i];
}
}  // namespace

std::vector<Route> route_experts(const Vec& logits, const Vec& correction, size_t k, float scale) {
    require(k && k <= logits.size() && logits.size() == correction.size(), "invalid router shape");
    Vec scores(logits.size()), choice(logits.size());
    std::vector<size_t> ids(logits.size());
    std::iota(ids.begin(), ids.end(), 0);
    for (size_t e : ids) {
        scores[e] = sigmoid(logits[e]);
        choice[e] = scores[e] + correction[e];
        require(std::isfinite(choice[e]), "non-finite routing score");
    }
    // The 708B has exactly one group. There is no group to mask out.
    std::partial_sort(ids.begin(), ids.begin() + k, ids.end(), [&](size_t a, size_t b) {
        return choice[a] == choice[b] ? a < b : choice[a] > choice[b];
    });
    ids.resize(k);
    std::sort(ids.begin(), ids.end());  // reference accumulates expert outputs in expert-ID order
    float denom = 1e-20f;
    for (size_t e : ids) denom += scores[e];
    std::vector<Route> result;
    for (size_t e : ids) result.push_back({e, (scores[e] / denom) * scale});
    return result;
}

Vec gated_norm(const Config& cfg, Weights& w, const std::string& p, const Vec& x) {
    const auto gamma = w.read(p + "weight", {x.size()});
    const float inv = 1.f / std::sqrt(std::inner_product(x.begin(), x.end(), x.begin(), 0.f) /
                                      float(x.size()) + cfg.at("rms_norm_eps").get<float>());
    Vec result(x.size());
    for (size_t i = 0; i < x.size(); ++i) result[i] = x[i] * inv * gamma[i];
    auto gate = w.linear(p + "gate_up_projection.weight", result, 16);
    for (auto& v : gate) v *= sigmoid(v);
    gate = w.linear(p + "gate_down_projection.weight", gate, x.size());
    const float factor = cfg.at("layernorm_gating_weight");
    for (size_t i = 0; i < x.size(); ++i) result[i] *= factor * sigmoid(gate[i]);
    return result;
}

Vec mlp(Weights& w, const std::string& p, const Vec& x, size_t intermediate, float clip) {
    auto gate = w.linear(p + "gate_proj.weight", x, intermediate);
    auto up = w.linear(p + "up_proj.weight", x, intermediate);
    for (size_t i = 0; i < intermediate; ++i) {
        const float g = clip > 0 ? std::min(gate[i], clip) : gate[i];
        const float u = clip > 0 ? std::clamp(up[i], -clip, clip) : up[i];
        up[i] = (g * sigmoid(g)) * u;
    }
    return w.linear(p + "down_proj.weight", up, x.size());
}

Decoder::Decoder(Config cfg, Weights& weights) : config_(std::move(cfg)), weights_(weights) {
    const auto& c = config_;
    require(c.at("model_type") == "gigachat4", "expected GigaChat4 checkpoint");
    require(c.at("norm_type") == "GatedNorm" && c.at("layernorm_type") == "pre_post" &&
            c.at("hidden_act") == "silu" && c.at("linear_attention_type") == "GigaChat4GatedDeltaNet",
            "unsupported norm, activation or attention variant");
    require(c.at("n_group") == 1 && c.at("topk_group") == 1 && c.at("norm_topk_prob") == true &&
            c.at("n_shared_experts") == 1 && c.at("use_shared_expert_sigmoid") == false,
            "expected 708B single-group router and ungated shared expert");
    require(c.at("dense_attention_layers").empty() && c.at("gated_attention") == true &&
            c.at("tie_word_embeddings") == false && !c.value("attention_bias", false),
            "expected sparse MLA, attention gate and untied head");
    if (c.contains("quantization_config") && !c["quantization_config"].is_null()) {
        const auto& q = c.at("quantization_config");
        require(q.at("quant_method") == "fp8" && q.at("fmt") == "e4m3" &&
                q.at("weight_block_size") == Config::array({128, 128}) &&
                q.at("activation_scheme") == "dynamic", "unsupported checkpoint quantization");
    }
    const auto count = dimension(c, "num_hidden_layers");
    require(c.at("first_k_dense_replace").is_number_integer() &&
            c.at("first_k_dense_replace").get<int64_t>() >= 0 &&
            c.at("first_k_dense_replace").get<size_t>() <= count, "invalid dense layer count");
    require(dimension(c, "num_experts_per_tok") <= dimension(c, "n_routed_experts"), "too many active experts");
    for (const char* key : {"hidden_size", "vocab_size", "intermediate_size", "moe_intermediate_size",
                            "max_position_embeddings"}) dimension(c, key);
    layers_.resize(count);
    sparse_.resize(count);
    for (const auto& id : c.at("full_attention_layers")) {
        require(id.is_number_integer() && id.get<size_t>() < count && !sparse_[id.get<size_t>()],
                "bad or duplicate attention layer");
        sparse_[id.get<size_t>()] = true;
    }
}

Vec Decoder::step(size_t token, Config* trace, Config* route_trace) {
    require(!failed_, "discard decoder after a failed step");
    const auto& c = config_;
    const size_t vocab = c.at("vocab_size"), hidden = c.at("hidden_size");
    require(token < vocab && position_ < c.at("max_position_embeddings").get<size_t>(), "token or position out of bounds");
    failed_ = true;  // a later failed read must not leave reusable partial layer state
    auto& w = weights_;
    Vec x = w.row("model.embed_tokens.weight", vocab, hidden, token);
    const float clip = c.at("swiglu_limit");
    for (size_t i = 0; i < layers_.size(); ++i) {
        const auto p = "model.layers." + std::to_string(i) + ".";
        Vec a = gated_norm(c, w, p + "input_layernorm.", x);
        auto& state = layers_[i];
        a = sparse_[i] ? dsa_step(c, w, p + "self_attn.", a, state, position_)
                       : gdn_step(c, w, p + "self_attn.", a, state);
        if (trace) {
            (*trace)["attention"].push_back(a);
            (*trace)["recurrent"].push_back(state.recurrent);
        }
        add(x, gated_norm(c, w, p + "post_self_attn_layernorm.", a));
        Vec h = gated_norm(c, w, p + "post_attention_layernorm.", x);
        Config ids = Config::array();
        if (i < c.at("first_k_dense_replace").get<size_t>()) {
            a = mlp(w, p + "mlp.", h, c.at("intermediate_size"), clip);
        } else {
            const size_t experts = c.at("n_routed_experts"), width = c.at("moe_intermediate_size");
            const auto r = route_experts(w.linear(p + "mlp.gate.weight", h, experts),
                                         w.read(p + "mlp.gate.e_score_correction_bias", {experts}),
                                         c.at("num_experts_per_tok"), c.at("routed_scaling_factor"));
            a.assign(hidden, 0.f);
            for (auto [expert, weight] : r) {
                if (trace || route_trace) ids.push_back(expert);
                add(a, mlp(w, p + "mlp.experts." + std::to_string(expert) + ".", h, width, clip), weight);
            }
            add(a, mlp(w, p + "mlp.shared_experts.", h, width, clip));
        }
        add(x, gated_norm(c, w, p + "post_feedforward_layernorm.", a));
        if (trace) {
            (*trace)["layers"].push_back(x);
            (*trace)["routes"].push_back(ids);
        }
        if (route_trace) route_trace->push_back(std::move(ids));
    }
    auto logits = w.linear("lm_head.weight", gated_norm(c, w, "model.norm.", x), vocab);
    const float cap = c.at("final_logit_softcapping");
    require(cap > 0, "unsupported logit cap");
    for (auto& v : logits) {
        require(std::isfinite(v), "non-finite logits");
        v = cap * std::tanh(v / cap);
    }
    if (trace) (*trace)["logits"] = logits;
    ++position_;
    failed_ = false;
    return logits;
}

}  // namespace strata::gigachat4
