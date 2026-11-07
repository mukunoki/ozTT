// SPDX-License-Identifier: Apache-2.0
#include <ttgemm/ttgemm.hpp>
#include <iostream>
#include <vector>

int main() {
    try {
        ttgemm::Options options;
        options.scheme = ttgemm::Scheme::II;
        options.format = ttgemm::Format::BFP8;
        if (!ttgemm::is_available(options.format))
            for (auto format : {ttgemm::Format::BF16, ttgemm::Format::FP16, ttgemm::Format::TF32, ttgemm::Format::FP8,
                                ttgemm::Format::INT8, ttgemm::Format::BFP4, ttgemm::Format::BFP2})
                if (ttgemm::is_available(format)) {
                    options.format = format;
                    break;
                }
        ttgemm::Context context;
        // 2x3 times 3x2; padding and tiling are handled by the library.
        const std::vector<double> a{1, 2, 3, 4, 5, 6};
        const std::vector<double> b{7, 8, 9, 10, 11, 12};
        const auto c = context.multiply(a, b, 2, 2, 3, options);
        std::cout << c[0] << ' ' << c[1] << '\n' << c[2] << ' ' << c[3] << '\n';
        // Expected: 58 64 / 139 154.
        return c == std::vector<double>{58, 64, 139, 154} ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
