#include "model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <queue>
#include <stdexcept>

namespace strata::gigachat4 {
namespace {

void require(bool ok, const std::string& why) {
    if (!ok) throw std::invalid_argument("GigaChat4 DSA: " + why);
}

size_t dim(const Config& c, const char* name, bool allow_zero = false) {
    const auto& v = c.at(name);
    require(v.is_number_integer() && (v.is_number_unsigned() || v.get<int64_t>() >= 0), name);
    auto n = v.get<uint64_t>();
    require((allow_zero || n > 0) && n <= Vec().max_size(), name);
    return static_cast<size_t>(n);
}

size_t mul(size_t a, size_t b) {
    require(!b || a <= Vec().max_size() / b, "shape product too large");
    return a * b;
}

size_t add(size_t a, size_t b) {
    require(a <= Vec().max_size() - b, "shape sum too large");
    return a + b;
}

float dot(const float* a, const float* b, size_t n) {
    return std::inner_product(a, a + n, b, 0.0f);
}

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

Vec vector_weight(Weights& w, const std::string& name, size_t n) {
    auto v = w.read(name, {n});
    require(v.size() == n, "wrong weight length: " + name);
    return v;
}

Vec project(Weights& w, const std::string& name, const Vec& x, size_t out, bool bias = false) {
    auto y = w.linear(name + ".weight", x, out);
    require(y.size() == out, "wrong projection length: " + name);
    if (bias) {
        auto b = vector_weight(w, name + ".bias", out);
        for (size_t i = 0; i < out; ++i) y[i] += b[i];
    }
    return y;
}

// GatedNorm acts on the learned, RMS-normalized latent, before MLA's alpha.
void norm(Weights& w, const std::string& name, const std::string& type, Vec& x, float eps, float gate_scale) {
    const bool gated = type == "GatedNorm" || type == "ZeroCenteredGatedNorm";
    const bool zero = type == "ZeroCenteredRMSNorm" || type == "ZeroCenteredGatedNorm";
    require(gated || zero || type == "LlamaRMSNorm", "unknown norm " + type);
    require(eps > 0 && std::isfinite(eps), "invalid norm epsilon");
    auto weight = vector_weight(w, name + ".weight", x.size());
    const float inv = 1.0f / std::sqrt(dot(x.data(), x.data(), x.size()) / x.size() + eps);
    for (size_t i = 0; i < x.size(); ++i) x[i] = x[i] * inv * (weight[i] + (zero ? 1.0f : 0.0f));
    if (gated) {
        Vec u = project(w, name + ".gate_up_projection", x, 16);
        for (float& e : u) e *= sigmoid(e);
        Vec g = project(w, name + ".gate_down_projection", u, x.size());
        for (size_t i = 0; i < x.size(); ++i) x[i] *= gate_scale * sigmoid(g[i]);
    }
}

double mscale(double factor, double m = 1) {
    return factor <= 1 ? 1.0 : 1.0 + 0.1 * m * std::log(factor);
}

double number_or(const Config& c, const char* key, double fallback) {
    auto it = c.find(key);
    return it == c.end() || it->is_null() ? fallback : it->get<double>();
}

struct Rotary {
    Vec cosine, sine, scratch;
    float logit_scale;

    Rotary(const Config& c, size_t width, size_t qk, size_t pos)
        : cosine(width / 2), sine(width / 2), scratch(width), logit_scale(1.0f / std::sqrt(float(qk))) {
        const Config empty = Config::object();
        const auto& r = c.contains("rope_scaling") && !c.at("rope_scaling").is_null() ? c.at("rope_scaling") : empty;
        const auto type = r.value("rope_type", r.value("type", std::string("default")));
        require(type == "default" || type == "yarn", "unsupported rope_type " + type);
        const double theta = r.value("rope_theta", c.value("rope_theta", 100000.0));
        require(std::isfinite(theta) && theta > 1, "invalid rope_theta");
        double factor = 1, mag = 1;
        float lo = 0, hi = 1;
        if (type == "yarn") {
            const double orig = r.value("original_max_position_embeddings", c.at("max_position_embeddings").get<double>());
            factor = r.at("factor").is_null() ? c.at("max_position_embeddings").get<double>() / orig : r.at("factor").get<double>();
            require(orig > 0 && std::isfinite(factor) && factor > 0, "invalid YaRN lengths");
            const double m = number_or(r, "mscale", 0.0), all = number_or(r, "mscale_all_dim", 0.0);
            mag = number_or(r, "attention_factor", m && all ? mscale(factor, m) / mscale(factor, all) : mscale(factor));
            logit_scale *= float(std::pow(mscale(factor, all), 2));
            auto correction = [&](const char* key, double fallback) {
                const double supplied = number_or(r, key, fallback);
                const double beta = supplied == 0 ? fallback : supplied;
                require(std::isfinite(beta) && beta > 0, "invalid YaRN beta");
                return width * std::log(orig / (beta * 6.2831853071795864769)) / (2 * std::log(theta));
            };
            double low = correction("beta_fast", 32), high = correction("beta_slow", 1);
            if (r.value("truncate", true)) { low = std::floor(low); high = std::ceil(high); }
            lo = float(std::max(0.0, low));
            hi = float(std::min(double(width - 1), high));
            if (lo == hi) hi += 0.001f;
        }
        for (size_t i = 0; i < cosine.size(); ++i) {
            // Match the transformers 5.8.1 fp32 inverse frequencies and angle multiply.
            const float f = std::pow(float(theta), float(2 * i) / float(width));
            const float extrap = 1 - std::clamp((float(i) - lo) / (hi - lo), 0.0f, 1.0f);
            const float base_inv = 1.0f / f, scaled_inv = 1.0f / (float(factor) * f);
            const float inv = type == "yarn" ? scaled_inv * (1 - extrap) + base_inv * extrap : base_inv;
            const float angle = float(pos) * inv;
            cosine[i] = std::cos(angle) * float(mag);
            sine[i] = std::sin(angle) * float(mag);
        }
    }

    // Interleave changes the INPUT pair layout. Both paths OUTPUT split halves.
    void apply(float* v, bool interleave) {
        const size_t half = cosine.size();
        std::copy_n(v, 2 * half, scratch.begin());
        for (size_t i = 0; i < half; ++i) {
            const float a = scratch[interleave ? 2*i : i], b = scratch[interleave ? 2*i+1 : half+i];
            v[i] = a * cosine[i] - b * sine[i];
            v[half+i] = b * cosine[i] + a * sine[i];
        }
    }
};

bool sparse_layer(const Config& c, const std::string& prefix) {
    if (!c.contains("index_topk") || c.at("index_topk").is_null()) return false;
    if (!c.contains("dense_attention_layers") || c.at("dense_attention_layers").empty()) return true;
    // The interface carries the layer id in the HF tensor prefix.
    const auto tag = prefix.rfind("layers.");
    require(tag != std::string::npos, "expected layers.<id>.self_attn. prefix");
    const auto start = tag + 7, end = prefix.find('.', start);
    require(end != std::string::npos && end > start && prefix.find_first_not_of("0123456789", start) == end &&
            prefix.substr(end) == ".self_attn.", "bad layer prefix");
    const auto layer = std::stoull(prefix.substr(start, end - start));
    const auto& dense = c.at("dense_attention_layers");
    return std::find(dense.begin(), dense.end(), layer) == dense.end();
}

// Reserve geometrically, but never allocate rows beyond the configured context.
void reserve_row(Vec& cache, size_t stride, size_t pos, size_t limit) {
    if (!stride) return;
    size_t rows = cache.capacity() / stride;
    if (pos < rows) return;
    const size_t grow = rows + std::min(rows, limit - rows);
    cache.reserve(mul(std::max(grow, pos + 1), stride));
}

struct Hit { float score; size_t pos; };
struct Better {
    bool operator()(Hit a, Hit b) const {
        return a.score == b.score ? a.pos < b.pos : a.score > b.score;
    }
};

} // namespace

Vec dsa_step(const Config& c, Weights& w, const std::string& p, const Vec& x, LayerState& s, size_t pos) {
    const size_t hidden = dim(c, "hidden_size"), heads = dim(c, "num_attention_heads");
    const size_t rank = dim(c, "kv_lora_rank"), rope = dim(c, "qk_rope_head_dim");
    const size_t nope = dim(c, "qk_nope_head_dim", true), value = dim(c, "v_head_dim");
    const size_t qk = add(nope, rope), nk = add(mul(heads, nope), rope), nv = mul(heads, value);
    const bool sparse = sparse_layer(c, p);
    const size_t idim = sparse ? dim(c, "index_head_dim") : 0;
    const size_t ih = sparse ? dim(c, "index_n_heads") : 0, topk = sparse ? dim(c, "index_topk") : 0;
    const size_t max_pos = dim(c, "max_position_embeddings");
    const size_t limit = c.contains("max_context") ? std::min(max_pos, dim(c, "max_context")) : max_pos;
    require(x.size() == hidden && !(rope % 2) && (!sparse || idim >= rope), "incompatible input or rotary dimensions");
    require(!c.contains("num_key_value_heads") || c.at("num_key_value_heads").is_null() ||
            dim(c, "num_key_value_heads") == heads, "MLA requires one expanded K/V per query head");
    require(pos < limit, "context limit exceeded");
    require(s.keys.size() == mul(pos, nk) && s.values.size() == mul(pos, nv) &&
            s.index_keys.size() == mul(pos, idim), "position does not match layer cache");
    const bool bias = c.value("attention_bias", false), scaled = c.value("use_mla_scaling_factor", true);
    const bool interleave = c.value("rope_interleave", true);
    const float eps = c.value("rms_norm_eps", 1e-6f), gs = c.value("layernorm_gating_weight", 2.f);
    const auto norm_type = c.value("norm_type", std::string("GatedNorm"));
    Rotary rot(c, rope, qk, pos);

    Vec latent, q;
    const Vec* q_input = &x;
    if (c.at("q_lora_rank").is_null()) {
        q = project(w, p + "q_proj", x, mul(heads, qk));
    } else {
        const size_t qr = dim(c, "q_lora_rank");
        latent = project(w, p + "q_a_proj", x, qr, bias);
        norm(w, p + "q_a_layernorm", norm_type, latent, eps, gs);
        const float alpha = scaled ? std::sqrt(float(hidden) / qr) : 1.f;
        for (float& v : latent) v *= alpha;
        q_input = &latent;
        q = project(w, p + "q_b_proj", latent, mul(heads, qk));
    }
    for (size_t h = 0; h < heads; ++h) rot.apply(q.data() + h*qk + nope, interleave);

    Vec kv = project(w, p + "kv_a_proj_with_mqa", x, add(rank, rope), bias);
    Vec k(nk), v(nv);
    std::copy(kv.begin() + rank, kv.end(), k.end() - rope); // shared rotary tail, once per token
    rot.apply(k.data() + nk - rope, interleave);
    kv.resize(rank);
    norm(w, p + "kv_a_layernorm", norm_type, kv, eps, gs);
    const float alpha_kv = scaled ? std::sqrt(float(hidden) / rank) : 1.f;
    for (float& e : kv) e *= alpha_kv;
    const size_t block = add(nope, value);
    kv = project(w, p + "kv_b_proj", kv, mul(heads, block));
    for (size_t h = 0; h < heads; ++h) {
        std::copy_n(kv.data() + h*block, nope, k.data() + h*nope);
        std::copy_n(kv.data() + h*block + nope, value, v.data() + h*value);
    }

    Vec ik;
    if (sparse) {
        ik = project(w, p + "indexer.wk", x, idim);
        const auto idx_type = c.contains("indexer_norm_type") && !c.at("indexer_norm_type").is_null()
                                  ? c.at("indexer_norm_type").get<std::string>() : "LlamaRMSNorm";
        const float idx_eps = float(number_or(c, "indexer_norm_eps", 1e-6));
        norm(w, p + "indexer.k_norm", idx_type.empty() ? "LlamaRMSNorm" : idx_type, ik,
             idx_eps == 0.f ? 1e-6f : idx_eps, 2.0f);
        rot.apply(ik.data(), false);
    }
    // Read the current token from local storage until o_proj succeeds. No future
    // token is present, even if this step is part of a larger prefill batch.
    auto row = [pos](const Vec& past, const Vec& now, size_t i) {
        return i == pos ? now.data() : past.data() + i * now.size();
    };
    std::vector<size_t> selected;
    if (sparse && pos + 1 > topk) {
        Vec iq = project(w, p + "indexer.wq_b", *q_input, mul(ih, idim));
        Vec hw = project(w, p + "indexer.weights_proj", x, ih);
        const float h_scale = 1.f / std::sqrt(float(ih)), d_scale = 1.f / std::sqrt(float(idim));
        for (size_t h = 0; h < ih; ++h) {
            rot.apply(iq.data() + h*idim, false);
            hw[h] *= h_scale;
        }
        std::priority_queue<Hit, std::vector<Hit>, Better> heap;
        for (size_t i = 0; i <= pos; ++i) {
            const float* key = row(s.index_keys, ik, i);
            float sum = 0;
            for (size_t h = 0; h < ih; ++h) sum += hw[h] * std::max(0.f, dot(iq.data() + h*idim, key, idim));
            const Hit hit{sum * d_scale, i};
            if (heap.size() < topk) heap.push(hit);
            else if (Better{}(hit, heap.top())) { heap.pop(); heap.push(hit); }
        }
        selected.reserve(topk);
        while (!heap.empty()) { selected.push_back(heap.top().pos); heap.pop(); }
        std::sort(selected.begin(), selected.end()); // natural key order for summation
    }
    const size_t count = selected.empty() ? pos + 1 : selected.size();
    auto index = [&](size_t i) { return selected.empty() ? i : selected[i]; };
    Vec out(nv, 0), scores(count);
    for (size_t h = 0; h < heads; ++h) {
        const float* query = q.data() + h*qk;
        for (size_t i = 0; i < count; ++i) {
            const float* key = row(s.keys, k, index(i));
            scores[i] = (dot(query, key + h*nope, nope) + dot(query + nope, key + nk - rope, rope)) * rot.logit_scale;
        }
        const float mx = *std::max_element(scores.begin(), scores.end());
        float den = 0;
        for (float& score : scores) { score = std::exp(score - mx); den += score; }
        for (size_t i = 0; i < count; ++i) {
            const float* val = row(s.values, v, index(i)) + h*value;
            const float prob = scores[i] / den;
            for (size_t j = 0; j < value; ++j) out[h*value+j] += prob * val[j];
        }
    }
    if (c.value("gated_attention", true)) {
        Vec gate = project(w, p + "gate_proj", x, nv);
        for (size_t j = 0; j < nv; ++j) out[j] *= sigmoid(gate[j]);
    }
    Vec result = project(w, p + "o_proj", out, hidden, bias);
    reserve_row(s.keys, nk, pos, limit);
    reserve_row(s.values, nv, pos, limit);
    reserve_row(s.index_keys, idim, pos, limit);
    s.keys.insert(s.keys.end(), k.begin(), k.end());
    s.values.insert(s.values.end(), v.begin(), v.end());
    s.index_keys.insert(s.index_keys.end(), ik.begin(), ik.end());
    return result;
}

} // namespace strata::gigachat4
