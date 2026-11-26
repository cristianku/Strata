#include "model.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <stdexcept>

using namespace strata::gigachat4;
namespace {

const std::string prefix = "model.layers.0.self_attn.";
void check(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
}
void close(float x, float y, const std::string& what, float tol = 2e-5f) {
    check(std::abs(x-y) <= tol, what + ": got " + std::to_string(x) + ", expected " + std::to_string(y));
}
template<typename F> void rejects(F fn, const char* what) {
    try { fn(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error(what);
}

// Exact binary fractions. The Python fixture fills named_parameters() in row-major
// order with the same sequence, so the test needs neither weights nor Python.
float sample(const std::string& name, size_t i) {
    uint32_t u = 2166136261u;
    for (unsigned char b : name) u = (u ^ b) * 16777619u;
    u += 0x9e3779b9u * uint32_t(i+1);
    u = (u ^ (u >> 16)) * 0x85ebca6bu;
    u = (u ^ (u >> 13)) * 0xc2b2ae35u;
    u ^= u >> 16;
    return (int(u >> 24)-128) / 256.f;
}
struct SmallWeights : Weights {
    Vec linear(const std::string& name, const Vec& x, size_t out) override {
        check(name.compare(0, prefix.size(), prefix) == 0, "projection prefix");
        std::string n = name.substr(prefix.size());
        Vec y(out, 0);
        for (size_t i = 0; i < out; ++i)
            for (size_t j = 0; j < x.size(); ++j) y[i] += sample(n, i*x.size()+j) * x[j];
        return y;
    }
    Vec read(const std::string& name, const std::vector<size_t>& shape) override {
        check(shape.size() == 1 && name.compare(0, prefix.size(), prefix) == 0, "read of non-vector");
        auto n = name.substr(prefix.size());
        Vec y(shape[0]);
        for (size_t i = 0; i < y.size(); ++i)
            y[i] = sample(n, i) + (n.find("norm.weight") != std::string::npos ? 1.f : 0.f);
        return y;
    }
    Vec row(const std::string&, size_t, size_t, size_t) override { throw std::logic_error("unexpected row"); }
};

Config small() {
    return {{"hidden_size",8}, {"num_attention_heads",2}, {"num_key_value_heads",2},
            {"q_lora_rank",5}, {"kv_lora_rank",3}, {"qk_rope_head_dim",8}, {"qk_nope_head_dim",3},
            {"v_head_dim",4}, {"index_n_heads",3}, {"index_head_dim",10}, {"index_topk",2},
            {"max_position_embeddings",256}, {"layernorm_gating_weight",1.8}, {"rope_theta",100.0},
            {"rope_scaling", {{"type","yarn"}, {"factor",4.0}, {"original_max_position_embeddings",64},
                              {"beta_fast",32}, {"beta_slow",1}, {"mscale",1.5}, {"mscale_all_dim",0.75}}}};
}
Vec input(size_t t) {
    Vec x(8);
    for (size_t j = 0; j < x.size(); ++j) x[j] = 2 * sample("input", t*x.size()+j);
    return x;
}

// Generated with the pinned HF GigaChat4Attention.eval().float(), an explicit
// triangular additive mask and positions 0..8 in ONE eager forward. See DSA.md.
constexpr float sparse_gold[9][8] = {
    {0.257094264f,0.075116552f,-0.152427971f,-0.293607771f,0.115618683f,-0.244459301f,-0.228138611f,0.229701087f},
    {0.578122795f,0.169109747f,-0.477491230f,-0.293488175f,0.268363059f,0.251453578f,-0.432043523f,0.463772297f},
    {0.412266731f,0.188166842f,-0.364316791f,-0.057598222f,0.107719034f,0.281400084f,-0.282702208f,0.496617228f},
    {0.566657782f,0.046462867f,-0.272190571f,-0.000993885f,0.105246425f,0.037509620f,-0.123096541f,0.761912286f},
    {0.254742980f,0.075115591f,-0.215026394f,-0.038514413f,0.103724480f,0.172020569f,-0.114834011f,0.255396873f},
    {-0.160597533f,0.008055057f,0.280184895f,0.345822603f,-0.416676939f,-0.256294012f,0.235472307f,0.275788248f},
    {-0.105633721f,0.027009567f,0.216403916f,0.248489603f,-0.346584886f,-0.244960502f,0.156765163f,0.252064586f},
    {0.198512539f,0.054399930f,-0.128591314f,-0.092992537f,0.073374018f,0.010689116f,-0.113536119f,0.230577052f},
    {-0.000736671f,-0.062848657f,0.085473955f,0.242245227f,-0.131756529f,0.035137348f,0.169994652f,0.135197774f},
};

void causal_parity() {
    Config c = small();
    SmallWeights w;
    LayerState state;
    for (size_t t = 0; t < 9; ++t) {
        auto out = dsa_step(c, w, prefix, input(t), state, t);
        check(out.size() == 8, "hidden size");
        for (size_t d = 0; d < 8; ++d) close(out[d], sparse_gold[t][d], "batched DSA oracle t=" + std::to_string(t));
    }
    const Vec ik{-2.064651728f,1.241879940f,-0.205539733f,-0.013144530f,0.630662441f,
                 2.077378273f,-0.258727640f,-0.851345003f,0.831476331f,0.187611416f};
    const Vec kr{0.040413901f,0.243553668f,-0.271400303f,-0.043497697f,-0.107729532f,
                 0.030983355f,0.016564138f,0.884434044f};
    check(state.index_keys.size() == 9*10 && state.keys.size() == 9*(2*3+8) && state.values.size() == 9*2*4,
          "KV layout: full history, one rotary tail / index key per token");
    for (size_t j = 0; j < ik.size(); ++j) close(state.index_keys[80+j], ik[j], "non-interleaved index cache");
    for (size_t j = 0; j < kr.size(); ++j) close(*(state.keys.end()-8+j), kr[j], "interleaved MLA cache");

    // HF can exclude individual full-attention layers from indexing.
    c.update(Config{{"dense_attention_layers", {0}}, {"q_lora_rank", nullptr}, {"rope_scaling", nullptr},
                    {"rope_interleave",false}, {"norm_type","LlamaRMSNorm"}, {"use_mla_scaling_factor",false},
                    {"gated_attention",false}, {"attention_bias",true}});
    state = {};
    Vec out;
    for (size_t t = 0; t < 9; ++t) out = dsa_step(c, w, prefix, input(t), state, t);
    const Vec dense_last{-0.330592364f,0.107840195f,0.681436062f,0.569931388f,-0.925864935f,-0.724096000f,0.346183419f,0.646958649f};
    check(state.index_keys.empty(), "dense layer acquired index cache");
    for (size_t d = 0; d < 8; ++d) close(out[d], dense_last[d], "batched dense MLA oracle");
}

void bounds() {
    auto c = small();
    SmallWeights w;
    LayerState state;
    c["max_context"] = 3;
    rejects([&]{ dsa_step(c,w,prefix,input(0),state,1); }, "skipped position accepted");
    for (size_t t = 0; t < 3; ++t) dsa_step(c,w,prefix,input(t),state,t);
    rejects([&]{ dsa_step(c,w,prefix,input(0),state,1); }, "rewind accepted");
    rejects([&]{ dsa_step(c,w,prefix,input(3),state,3); }, "context overflow accepted");
    check(state.keys.capacity() == 3*14 && state.values.capacity() == 3*8 && state.index_keys.capacity() == 3*10,
          "reserve beyond requested context");
    c["max_position_embeddings"] = 3; c["max_context"] = 8;
    rejects([&]{ dsa_step(c,w,prefix,input(3),state,3); }, "model context overflow accepted");
    c["max_position_embeddings"] = 8;
    state.index_keys.pop_back();
    rejects([&]{ dsa_step(c,w,prefix,input(3),state,3); }, "desynchronized index cache accepted");
}

Config published() {
    return {{"hidden_size",7168}, {"num_attention_heads",64}, {"num_key_value_heads",64},
            {"q_lora_rank",1536}, {"kv_lora_rank",512}, {"qk_rope_head_dim",64}, {"qk_nope_head_dim",128},
            {"v_head_dim",128}, {"index_n_heads",64}, {"index_head_dim",128}, {"index_topk",2048},
            {"max_position_embeddings",1048576}, {"rope_theta",100000.0},
            {"rope_scaling",{{"type","yarn"}, {"factor",32.0}, {"original_max_position_embeddings",32768},
                             {"beta_fast",32}, {"beta_slow",1}, {"mscale",1.0}, {"mscale_all_dim",1.0}}}};
}

// These [out,in] shapes match model.layers.3.self_attn in the published FP8
// Safetensors headers. No [out,in] matrix for the 708B is allocated/read here.
struct ShapeWeights : SmallWeights {
    std::map<std::string, std::pair<size_t,size_t>> matrices{
        {"q_a_proj",{1536,7168}}, {"q_b_proj",{12288,1536}}, {"kv_a_proj_with_mqa",{576,7168}},
        {"kv_b_proj",{16384,512}}, {"o_proj",{7168,8192}}, {"gate_proj",{8192,7168}},
        {"q_a_layernorm.gate_up_projection",{16,1536}}, {"q_a_layernorm.gate_down_projection",{1536,16}},
        {"kv_a_layernorm.gate_up_projection",{16,512}}, {"kv_a_layernorm.gate_down_projection",{512,16}},
        {"indexer.wq_b",{8192,1536}}, {"indexer.wk",{128,7168}}, {"indexer.weights_proj",{64,7168}},
    };
    std::map<std::string, size_t> reads{
        {"q_a_layernorm.weight",1536}, {"kv_a_layernorm.weight",512}, {"indexer.k_norm.weight",128}
    };
    std::map<std::string,size_t> calls;
    Vec linear(const std::string& name, const Vec& x, size_t out) override {
        const auto n = name.substr(prefix.size(), name.size()-prefix.size()-7);
        check(name == prefix+n+".weight" && matrices.at(n) == std::make_pair(out,x.size()), "708B header " + name);
        ++calls[n];
        return Vec(out, 0);
    }
    Vec read(const std::string& name, const std::vector<size_t>& shape) override {
        check(shape == std::vector<size_t>{reads.at(name.substr(prefix.size()))}, "708B norm header " + name);
        return Vec(shape[0], 1);
    }
};

void shape_contract() {
    auto c = published();
    ShapeWeights w;
    LayerState s;
    Vec x(7168,1.f);
    dsa_step(c,w,prefix,x,s,0);
    c["index_topk"] = 1; // exercise both indexer query projections without a 2049-token prefill
    dsa_step(c,w,prefix,x,s,1);
    check(w.calls.size() == w.matrices.size(), "untested 708B projection");
    check(s.keys.size() == 2*(8192+64) && s.values.size() == 2*8192 && s.index_keys.size() == 2*128,
          "unexpected 708B cache size");
}

struct RampWeights : SmallWeights {
    Vec linear(const std::string& n, const Vec&, size_t out) override {
        Vec r(out);
        if (n == prefix + "kv_a_proj_with_mqa.weight")
            for (size_t i = 0; i < 64; ++i) r[1+i] = float(i+1)/64;
        return r;
    }
};
void long_position() {
    auto c = published();
    c.update(Config{{"num_attention_heads",1}, {"num_key_value_heads",1}, {"hidden_size",1},
                    {"q_lora_rank",nullptr}, {"kv_lora_rank",1}, {"qk_nope_head_dim",0},
                    {"v_head_dim",1}, {"index_topk",nullptr}, {"gated_attention",false}, {"norm_type","LlamaRMSNorm"}});
    // Only the rotary tail matters. Exercise the actual 708B-width frequency
    // ramp at the original training length, using zero-filled history.
    const size_t t = 32768;
    LayerState s;
    s.keys.resize(64*t); s.values.resize(t);
    RampWeights w;
    dsa_step(c,w,prefix,{1.f},s,t);
    const size_t columns[] = {0,7,15,16,17,20,22,23,27,31,32,39,47,48,49,52,54,55,59,63};
    const float gold[] = {-.023168357f,-.266831070f,-.684731960f,-.657414973f,-.200394154f,
            .083906531f,-.945497632f,-.305185795f,.803619802f,.969595492f,.026152063f,.215018183f,
            .125543922f,.340442836f,.758498967f,-.913249254f,-.342082709f,1.004324794f,.926469445f,1.014336586f};
    for (size_t j = 0; j < sizeof(columns)/sizeof(columns[0]); ++j)
        close(s.keys[t*64+columns[j]], gold[j], "published YaRN at position 32768", 4e-5f);
}
} // namespace

int main() {
    try {
        causal_parity(); bounds(); shape_contract(); long_position();
        std::cout << "GigaChat4 DSA: CPU oracle, cache bounds, 708B projection shapes and long-position YaRN passed\n";
    } catch (const std::exception& e) {
        std::cerr << "GigaChat4 DSA test: " << e.what() << '\n'; return 1;
    }
}
