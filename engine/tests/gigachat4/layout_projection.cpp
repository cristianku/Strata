// Companion to check_layout.py --sample-projections. The fixture is three
// slices, not a full checkpoint. Exercise the ordinary native CPU file reader.
#include "weights.hpp"

#include <exception>
#include <iostream>

namespace gc = strata::gigachat4;

int main(int argc, char** argv) try {
    if (argc != 2) return 2;
    gc::FileWeights weights(argv[1]);
    gc::Vec x(7168);
    for (size_t i = 0; i < x.size(); ++i) x[i] = (int(i % 17) - 8) / 8.0f;
    auto y = weights.linear("model.layers.3.self_attn.kv_a_proj_with_mqa.weight", x, 2);
    const auto z = weights.linear("model.layers.3.self_attn.indexer.weights_proj.weight", x, 1);
    y.insert(y.end(), z.begin(), z.end());
    std::cout << gc::Config(y) << '\n';
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
