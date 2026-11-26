#include "decoder.hpp"
#include "weights.hpp"
#ifdef STRATA_GIGACHAT4_CUDA
#include "cuda_backend.hpp"
#endif

#include <algorithm>
#include <charconv>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace gc = strata::gigachat4;
using Clock = std::chrono::steady_clock;
using Seconds = std::chrono::duration<double>;

namespace {
size_t number(const std::string& s) {
    size_t v;
    const auto r = std::from_chars(s.data(), s.data() + s.size(), v);
    if (r.ec != std::errc{} || r.ptr != s.data() + s.size())
        throw std::invalid_argument("expected non-negative integer: " + s);
    return v;
}

size_t mib(const std::string& s) {
    const size_t v = number(s);
    if (v > std::numeric_limits<size_t>::max() / (1u << 20)) throw std::invalid_argument("cache budget too large");
    return v * (1u << 20);
}

void json_file(const std::string& path, const gc::Config& j) {
    std::ofstream os(path);
    os << j << '\n';
    os.close();
    if (!os) throw std::runtime_error("cannot write " + path);
}

const char* usage =
    "strata-gigachat4 --model DIR --tokens ID[,ID...] [--generate N] [--ignore-eos]\n"
    "  [--host-cache-mib N] [--cuda --gpu-cache-mib N] [--prefetch] [--routes FILE.jsonl] [--trace FILE]\n"
    "  Prints generated IDs and timing counters as JSON. --trace is for tiny fixtures.\n";
}  // namespace

int main(int argc, char** argv) try {
    std::string dir, trace_path, routes_path;
    std::vector<size_t> tokens;
    size_t generate = 1, host = 0, gpu = 0;
    bool cuda = false, ignore_eos = false, prefetch = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() -> std::string {
            if (++i == argc) throw std::invalid_argument("missing value for " + arg);
            return argv[i];
        };
        if (arg == "--help") { std::cout << usage; return 0; }
        else if (arg == "--model") dir = value();
        else if (arg == "--tokens") {
            const auto s = value();
            size_t start = 0, end;
            do {
                end = s.find(',', start);
                tokens.push_back(number(s.substr(start, end - start)));
                if (end != std::string::npos) start = end + 1;
            } while (end != std::string::npos);
        } else if (arg == "--generate") generate = number(value());
        else if (arg == "--host-cache-mib") host = mib(value());
        else if (arg == "--gpu-cache-mib") gpu = mib(value());
        else if (arg == "--cuda") cuda = true;
        else if (arg == "--ignore-eos") ignore_eos = true;
        else if (arg == "--prefetch") prefetch = true;
        else if (arg == "--trace") trace_path = value();
        else if (arg == "--routes") routes_path = value();
        else throw std::invalid_argument("unknown option " + arg);
    }
    if (dir.empty() || tokens.empty() || (gpu && !cuda)) throw std::invalid_argument(usage);
    const auto start = Clock::now();
    std::ifstream config_file(std::filesystem::path(dir) / "config.json");
    if (!config_file) throw std::runtime_error("missing config.json");
    gc::Config cfg = gc::Config::parse(config_file);
    const size_t limit = cfg.at("max_position_embeddings");
    if (tokens.size() > limit || (generate && generate - 1 > limit - tokens.size()))
        throw std::invalid_argument("prompt plus decode exceeds model context");
    const size_t total_steps = tokens.size() + (generate ? generate - 1 : 0);
    cfg["max_context"] = total_steps; // limit cache capacity growth to this request
    if (!trace_path.empty() && (cfg.at("hidden_size").get<size_t>() > 512 || total_steps > 256))
        throw std::invalid_argument("--trace is for reduced fixtures; use --routes for full-size profiling");
    std::unique_ptr<gc::MatvecBackend> backend;
    if (cuda) {
#ifdef STRATA_GIGACHAT4_CUDA
        backend = gc::make_cuda_backend(gpu);
#else
        throw std::invalid_argument("this build does not include CUDA");
#endif
    }
    gc::FileWeights w(dir, host, backend.get());
    gc::Decoder decoder(cfg, w, prefetch);
    gc::Config steps = gc::Config::array();
    size_t position = 0;
    std::ofstream routes_file;
    if (!routes_path.empty()) {
        routes_file.exceptions(std::ios::failbit | std::ios::badbit);
        routes_file.open(routes_path);
    }
    auto step = [&](size_t token) {
        gc::Config trace, routes = gc::Config::array();
        auto logits = decoder.step(token, trace_path.empty() ? nullptr : &trace,
                                   routes_path.empty() ? nullptr : &routes);
        if (!trace_path.empty()) steps.push_back(std::move(trace));
        if (!routes_path.empty()) {
            for (size_t layer = 0; layer < routes.size(); ++layer)
                if (!routes[layer].empty())
                    routes_file << gc::Config{{"token", position}, {"layer", layer}, {"experts", routes[layer]}} << '\n';
        }
        ++position;
        return logits;
    };
    auto greedy = [](const gc::Vec& v) { return size_t(std::max_element(v.begin(), v.end()) - v.begin()); };
    std::vector<size_t> generated;
    const auto loaded = Clock::now();
    gc::Vec logits;
    for (size_t token : tokens) logits = step(token);
    if (generate) generated.push_back(greedy(logits));
    const auto first = Clock::now();
    const size_t eos = cfg.at("eos_token_id");
    while (generated.size() < generate && (ignore_eos || generated.back() != eos)) {
        generated.push_back(greedy(step(generated.back())));
    }
    const auto end = Clock::now();
    const auto s = w.stats();
    const gc::Config weights = {
        {"payload_read_ops", s.read_ops}, {"logical_payload_bytes_read", s.bytes_read},
        {"payload_read_seconds", s.payload_read_seconds},
        {"prefetch_enabled", prefetch}, {"prefetch_supported", s.prefetch_supported},
        {"prefetch_requests", s.prefetch_requests}, {"prefetch_bytes_advised", s.prefetch_bytes},
        {"prefetch_errors", s.prefetch_errors},
        {"host_cache_hits", s.cache_hits}, {"host_cache_misses", s.cache_misses}, {"evictions", s.evictions},
        {"peak_host_cache_bytes", s.peak_cache_bytes}, {"largest_read_bytes", s.max_read_bytes},
        {"tensor_count", s.tensors}, {"shard_count", s.shards}
    };
    gc::Config metrics = {
        {"load_seconds", Seconds(loaded-start).count()}, {"prefill_seconds", Seconds(first-loaded).count()},
        {"ttft_seconds", generate ? gc::Config(Seconds(first-start).count()) : gc::Config(nullptr)},
        {"decode_seconds", Seconds(end-first).count()}, {"decode_steps", generated.empty() ? 0 : generated.size()-1},
        {"prompt_tokens", tokens.size()}, {"weights", weights}, {"backend", cuda ? "cuda" : "cpu"}
    };
#ifdef STRATA_GIGACHAT4_CUDA
    if (cuda) metrics["cuda"] = gc::cuda_metrics(*backend);
#endif
    if (!trace_path.empty()) json_file(trace_path, {{"steps", std::move(steps)}, {"generated", generated}});
    if (!routes_path.empty()) routes_file.close();
    std::cout << gc::Config{{"generated", generated}, {"metrics", metrics}} << '\n';
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
