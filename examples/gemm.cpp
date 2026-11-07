// SPDX-License-Identifier: Apache-2.0
#include <ozTTgemm/ozTTgemm.hpp>
#include <iostream>
#include <vector>

int main() {
    try {
        ozTTgemm::Options options;
        options.scheme = ozTTgemm::Scheme::II;
        options.format = ozTTgemm::Format::BFP8;
        if (!ozTTgemm::is_available(options.format))
            for (auto format :
                 {ozTTgemm::Format::BF16, ozTTgemm::Format::FP16, ozTTgemm::Format::TF32, ozTTgemm::Format::FP8,
                  ozTTgemm::Format::INT8, ozTTgemm::Format::BFP4, ozTTgemm::Format::BFP2})
                if (ozTTgemm::is_available(format)) {
                    options.format = format;
                    break;
                }
        ozTTgemm::Context context;
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
