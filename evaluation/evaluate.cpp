// SPDX-License-Identifier: Apache-2.0
//
// Paper evaluator: runs the planned cases through the public API with the timing hooks
// of src/evaluation_hooks.hpp enabled, checks the result against a Dot2 reference and
// an optional CPU DGEMM, and writes one CSV row per case. See evaluation/README.md.
#include <ttgemm/ttgemm.hpp>
#include "evaluation_hooks.hpp"
#include "numerics.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace ev = ttgemm::detail::evaluation;
namespace {
// Command-line settings (defaults are the paper conditions).
struct Config {
    std::string suite = "all", output = "results.csv", blas;
    std::vector<unsigned> sizes{3520, 7040, 14080};
    unsigned phi_size = 14080, repeats = 3, device = 0;
    std::optional<unsigned> sysfs_device;  // telemetry sysfs index when it differs from the Metalium device ID
    std::optional<unsigned> only;          // run only this case of the plan (0-based) and append to the CSV
    std::vector<double> phis{0.5, 1, 2, 4, 8};
    double warmup = 10, cooldown = 60, timeout = 300;
    bool list = false, reference = true, cpu = true;
    std::vector<ttgemm::Format> formats{ttgemm::Format::BF16, ttgemm::Format::FP16, ttgemm::Format::TF32,
                                        ttgemm::Format::BFP8, ttgemm::Format::FP8,  ttgemm::Format::INT8,
                                        ttgemm::Format::BFP4, ttgemm::Format::BFP2};
};
// Comma-separated list; empty entries are rejected.
std::vector<std::string> split(const std::string& text) {
    std::vector<std::string> out;
    std::istringstream in(text);
    std::string v;
    while (std::getline(in, v, ',')) {
        if (v.empty()) throw std::invalid_argument("empty list entry");
        out.push_back(v);
    }
    if (out.empty() || text.back() == ',') throw std::invalid_argument("empty list");
    return out;
}
double number(const std::string& text, double low, double high) {
    std::size_t used = 0;
    const double v = std::stod(text, &used);
    if (used != text.size() || !std::isfinite(v) || v < low || v > high)
        throw std::invalid_argument("invalid numeric argument: " + text);
    return v;
}
// Matrix size: a multiple of 64 so that no padding enters the timing.
unsigned size(const std::string& text) {
    const double v = number(text, 64, 32768);
    if (std::floor(v) != v || unsigned(v) % 64) throw std::invalid_argument("sizes must be multiples of 64");
    return unsigned(v);
}
Config args(int argc, char** argv) {
    Config c;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto value = [&] {
            if (++i == argc) throw std::invalid_argument("missing value for " + key);
            return std::string(argv[i]);
        };
        if (key == "--suite")
            c.suite = value();
        else if (key == "--sizes") {
            c.sizes.clear();
            for (auto& s : split(value())) c.sizes.push_back(size(s));
        } else if (key == "--phi-size")
            c.phi_size = size(value());
        else if (key == "--phis") {
            c.phis.clear();
            for (auto& s : split(value())) c.phis.push_back(number(s, 0, 16));
        } else if (key == "--formats") {
            const auto all = c.formats;
            c.formats.clear();
            for (auto& name : split(value())) {
                bool found = false;
                for (auto f : all)
                    if (name == ttgemm::format_name(f)) {
                        c.formats.push_back(f);
                        found = true;
                        break;
                    }
                if (!found) throw std::invalid_argument("unknown format: " + name);
            }
        } else if (key == "--repeats") {
            const double v = number(value(), 1, 100);
            if (v != std::floor(v)) throw std::invalid_argument("repeats must be an integer");
            c.repeats = unsigned(v);
        } else if (key == "--device") {
            const double v = number(value(), 0, 1024);
            if (v != std::floor(v)) throw std::invalid_argument("device must be an integer");
            c.device = unsigned(v);
        } else if (key == "--sysfs-device") {
            const double v = number(value(), 0, 1024);
            if (v != std::floor(v)) throw std::invalid_argument("sysfs-device must be an integer");
            c.sysfs_device = unsigned(v);
        } else if (key == "--case") {
            const double v = number(value(), 0, 100000);
            if (v != std::floor(v)) throw std::invalid_argument("case must be an integer");
            c.only = unsigned(v);
        } else if (key == "--warmup")
            c.warmup = number(value(), 0, 180);
        else if (key == "--cooldown")
            c.cooldown = number(value(), 0, 100);
        else if (key == "--timeout")
            c.timeout = number(value(), 1, 3600);
        else if (key == "--output")
            c.output = value();
        else if (key == "--blas")
            c.blas = value();
        else if (key == "--list")
            c.list = true;
        else if (key == "--no-reference") {
            c.reference = false;
            c.cpu = false;
        } else if (key == "--no-cpu")
            c.cpu = false;
        else if (key == "--quick") {
            c.sizes = {128};
            c.phi_size = 128;
            c.phis = {0.5, 2, 8};
            c.repeats = 1;
            c.warmup = 0;
            c.cooldown = 0;
            c.cpu = false;
        } else
            throw std::invalid_argument("unknown argument: " + key);
    }
    if (c.suite != "table" && c.suite != "phi" && c.suite != "all")
        throw std::invalid_argument("suite must be table, phi or all");
    if (c.suite != "phi")
        for (auto f : c.formats)
            if (!ttgemm::is_available(f)) throw std::invalid_argument("a requested table format was not built");
    if (c.suite != "table" && !ttgemm::is_available(ttgemm::Format::BFP8))
        throw std::invalid_argument("phi evaluation requires BFP8");
    return c;
}
// One evaluation: suite ("table" or "phi"), size, phi (negative for the uniform input
// of the table suite), the library options and the fixed b of scheme II (0: automatic).
struct Case {
    std::string suite;
    unsigned n;
    double phi;
    ttgemm::Options options;
    unsigned bits;
};
// The evaluation plan in its fixed order; --case indexes into this list.
std::vector<Case> cases(const Config& c) {
    constexpr unsigned slices[]{10, 10, 10, 10, 12, 8, 16, 59}, moduli[]{24, 24, 11, 24, 14, 16, 24, 24};
    std::vector<Case> out;
    if (c.suite != "phi")
        for (auto n : c.sizes)
            for (auto f : c.formats)
                for (auto scheme : {ttgemm::Scheme::I, ttgemm::Scheme::II}) {
                    ttgemm::Options o;
                    o.format = f;
                    o.scheme = scheme;
                    if (scheme == ttgemm::Scheme::I)
                        o.slices = slices[unsigned(f)];
                    else {
                        o.max_moduli = moduli[unsigned(f)];
                        o.reconstruction = ttgemm::Reconstruction::OnePass;
                    }
                    out.push_back({"table", n, -1, o, scheme == ttgemm::Scheme::II ? 54u : 0u});
                }
    if (c.suite != "table")
        for (auto phi : c.phis) {
            ttgemm::Options o;
            o.format = ttgemm::Format::BFP8;
            o.scheme = ttgemm::Scheme::I;
            out.push_back({"phi", c.phi_size, phi, o, 0});
            o.scheme = ttgemm::Scheme::II;
            o.max_moduli = 24;
            o.reconstruction = ttgemm::Reconstruction::OnePass;
            out.push_back({"phi", c.phi_size, phi, o, 0});
            o.reconstruction = ttgemm::Reconstruction::TwoPass;
            out.push_back({"phi", c.phi_size, phi, o, 0});
        }
    return out;
}
// Device telemetry from sysfs: temperature, and the AI clock sampled by a background
// thread so that the mean clock of a timed interval can be reported.
class Monitor {
    struct Sample {
        double t, clock;
    };
    std::filesystem::path clock_, temperature_;
    mutable std::mutex mutex_;
    std::vector<Sample> samples_;
    static double read(const std::filesystem::path& path) {
        double v;
        std::ifstream f(path);
        return f >> v ? v : std::numeric_limits<double>::quiet_NaN();
    }
    std::jthread worker_;

public:
    explicit Monitor(unsigned device, bool required) {
        const auto root = std::filesystem::path("/sys/class/tenstorrent") / ("tenstorrent!" + std::to_string(device));
        clock_ = root / "tt_aiclk";
        std::error_code error;
        for (const auto& item : std::filesystem::directory_iterator(root / "device/hwmon", error))
            if (std::filesystem::is_regular_file(item.path() / "temp1_input")) {
                temperature_ = item.path() / "temp1_input";
                break;
            }
        if (required && (!std::isfinite(read(clock_)) || !std::isfinite(temperature())))
            throw std::runtime_error("device clock/temperature sysfs readings are required for a paper run");
        worker_ = std::jthread([this](std::stop_token stop) {
            while (!stop.stop_requested()) {
                const Sample s{ev::now(), read(clock_)};
                {
                    std::lock_guard lock(mutex_);
                    samples_.push_back(s);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        });
    }
    double temperature() const { return read(temperature_) / 1000; }
    double clock(double begin, double end) const {
        std::lock_guard lock(mutex_);
        if (samples_.empty() || samples_.front().t > begin || samples_.back().t + 0.1 < end) return NAN;
        double sum = 0, weight = 0;
        for (std::size_t i = 0; i < samples_.size(); ++i) {
            const auto& s = samples_[i];
            const double next = i + 1 < samples_.size() ? samples_[i + 1].t : end;
            const double dt = std::min(end, next) - std::max(begin, s.t);
            if (dt <= 0) continue;
            if (!std::isfinite(s.clock) || next - s.t > 0.1) return NAN;
            sum += dt * s.clock;
            weight += dt;
        }
        return weight > 0 ? sum / weight : NAN;
    }
    void cool(double limit, double timeout) {
        if (limit == 0) return;
        const double begin = ev::now();
        for (;;) {
            const double value = temperature();
            if (!std::isfinite(value)) throw std::runtime_error("temperature reading failed during cooldown");
            if (value <= limit) return;
            if (ev::now() - begin >= timeout) throw std::runtime_error("device did not cool before the timeout");
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
};
// Warmup policy passed to the library hooks: a group of device steps is repeated until
// the clock has settled, and again if the group has been idle since it was last warm.
class Warmup {
    Monitor& monitor_;
    double minimum_;
    struct Group {
        bool warm = false;
        double last = 0;
    };
    std::map<std::string, Group> groups_;

public:
    Warmup(Monitor& m, double minimum) : monitor_(m), minimum_(minimum) {}
    // The two compared windows are aligned to the starts of issued units, as in the
    // paper harness: the last window runs from the latest start at least one second
    // before now, the previous one from the latest start at least one second before
    // that. Unaligned windows would compare different mixes of a unit's phases.
    static bool windows(const std::vector<double>& starts, double now, double& a, double& m) {
        auto it = std::upper_bound(starts.begin(), starts.end(), now - 1.0);
        if (it == starts.begin()) return false;
        m = *(it - 1);
        auto it2 = std::upper_bound(starts.begin(), starts.end(), m - 1.0);
        if (it2 == starts.begin()) return false;
        a = *(it2 - 1);
        return true;
    }
    void operator()(std::string_view name, const std::function<void()>& issue) {
        if (minimum_ == 0) return;
        auto& g = groups_[std::string(name)];
        const double begin = ev::now(), gap = g.warm ? begin - g.last : 0;
        // As in the paper harness: a gap above 50 ms requires a new steady
        // clock check; short gaps are covered by 20 times the gap in execution.
        const bool full = !g.warm, steady = full || gap > 0.05;
        std::vector<double> starts;
        double checked = -1;
        for (;;) {
            starts.push_back(ev::now());
            issue();
            const double now = ev::now();
            if (steady) {
                if (now - begin >= std::max(full ? minimum_ : 0.0, 2.0) &&
                    (now - checked >= 0.05 || now - begin >= 180)) {
                    checked = now;
                    double a, m;
                    if (windows(starts, now, a, m)) {
                        const double previous = monitor_.clock(a, m), last = monitor_.clock(m, now);
                        if (std::isfinite(previous) && previous > 0 && std::isfinite(last) &&
                            std::abs(last - previous) / previous < 0.01)
                            break;
                    }
                }
            } else if (now - begin >= 20 * gap)
                break;
            if (now - begin >= 180) throw std::runtime_error("AICLK did not settle during phase warmup");
        }
        g.warm = true;
        g.last = ev::now();
    }
    void measured(std::string_view name, double end) { groups_[std::string(name)].last = end; }
};
// CPU DGEMM for the FP64-equivalence criterion, loaded from an LP64 OpenBLAS at run time.
class Blas {
    struct Close {
        void operator()(void* handle) const { dlclose(handle); }
    };
    std::unique_ptr<void, Close> handle_;
    using Gemm = void (*)(int, int, int, int, int, int, double, const double*, int, const double*, int, double, double*,
                          int);
    Gemm gemm_ = nullptr;

public:
    explicit Blas(const Config& c) {
        if (!c.cpu) return;
        handle_.reset(dlopen(c.blas.empty() ? "libopenblas.so.0" : c.blas.c_str(), RTLD_NOW | RTLD_LOCAL));
        if (!handle_)
            throw std::runtime_error(
                std::string(
                    "OpenBLAS is required for the CPU comparison; install libopenblas0, use --blas PATH, or use --no-cpu: ") +
                dlerror());
        gemm_ = reinterpret_cast<Gemm>(dlsym(handle_.get(), "cblas_dgemm"));
        const auto config = reinterpret_cast<const char* (*)()>(dlsym(handle_.get(), "openblas_get_config"));
        if (!gemm_ || !config || std::string(config()).find("USE64BITINT") != std::string::npos)
            throw std::runtime_error("an LP64 OpenBLAS with cblas_dgemm is required");
        std::cout << "OpenBLAS: " << config() << '\n';
    }
    double bits(const std::vector<double>& a, const std::vector<double>& b, const std::vector<double>& hi,
                const std::vector<double>& lo, unsigned n, std::vector<double>& c) {
        if (!gemm_) return NAN;
        gemm_(101, 111, 111, n, n, n, 1, a.data(), n, b.data(), n, 0, c.data(), n);
        return paper_eval::accuracy(c, hi, lo).ebits;
    }
};
const char* scheme(ttgemm::Scheme s) { return s == ttgemm::Scheme::I ? "I" : "II"; }
// Requested reconstruction passes: 0 for scheme I, "auto" when the library selects them.
const char* passes(const ttgemm::Options& o) {
    if (o.scheme == ttgemm::Scheme::I) return "0";
    switch (o.reconstruction) {
    case ttgemm::Reconstruction::OnePass: return "1";
    case ttgemm::Reconstruction::TwoPass: return "2";
    default: return "auto";
    }
}
void list(const std::vector<Case>& plan) {
    std::cout << "suite,n,scheme,format,phi,slices,max_moduli,fixed_bits,passes\n";
    for (const auto& c : plan)
        std::cout << c.suite << ',' << c.n << ',' << scheme(c.options.scheme) << ','
                  << ttgemm::format_name(c.options.format) << ',' << c.phi << ',' << c.options.slices << ','
                  << c.options.max_moduli << ',' << c.bits << ',' << passes(c.options) << '\n';
}
}  // namespace
int main(int argc, char** argv) {
    try {
        const Config cfg = args(argc, argv);
        auto plan = cases(cfg);
        if (cfg.list) {
            list(plan);
            return 0;
        }
        // One case per process (evaluate.sh): the device cools at its idle clock only
        // after the process that opened it has exited. Later cases append to the CSV.
        const bool append = cfg.only && *cfg.only > 0;
        if (cfg.only) {
            if (*cfg.only >= plan.size()) throw std::invalid_argument("case index is outside the plan");
            plan = {plan[*cfg.only]};
        }
        Blas blas(cfg);
        const auto path = std::filesystem::path(cfg.output);
        if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
        std::ofstream csv(path, append ? std::ios::app : std::ios::trunc);
        if (!csv) throw std::runtime_error("cannot open output CSV");
        csv.exceptions(std::ios::badbit | std::ios::failbit);
        if (!append)
            csv << "suite,n,scheme,format,phi,timing,reference,slices,integer_bits,moduli,passes,partial_products,bw,scale_ms,slice_ms,gemm_ms,compose_ms,recon_ms,phase_sum_ms,total_ms,effective_tflops,gemm_tops,nominal_peak_percent,gemm_aiclk_mhz,evaluation_wall_ms,effective_bits,relative_fro,output_hash,cpu_dgemm_bits,fp64_equivalent,start_temperature_c\n";
        csv << std::setprecision(12);
        Monitor monitor(cfg.sysfs_device.value_or(cfg.device), cfg.warmup > 0 || cfg.cooldown > 0);
        ttgemm::DeviceOptions device;
        device.device_id = cfg.device;
        ttgemm::Context context(device);
        std::vector<double> a, b, hi, lo, c;
        unsigned current_n = 0;
        double current_phi = -2, cpu_bits = NAN;
        for (const auto& task : plan) {
            if (task.n != current_n || task.phi != current_phi) {
                std::cout << "Generating input and Dot2 reference: n=" << task.n << " phi=" << task.phi << std::endl;
                paper_eval::inputs(a, b, task.n, task.phi);
                c.resize(a.size());
                if (cfg.reference) paper_eval::reference_gemm(a, b, task.n, task.n, task.n, hi, lo);
                cpu_bits = cfg.reference ? blas.bits(a, b, hi, lo, task.n, c) : NAN;
                current_n = task.n;
                current_phi = task.phi;
            }
            std::cout << "Evaluating " << task.suite << ' ' << task.n << ' ' << scheme(task.options.scheme) << '-'
                      << ttgemm::format_name(task.options.format) << " phi=" << task.phi
                      << " passes=" << passes(task.options) << std::endl;
            monitor.cool(cfg.cooldown, cfg.timeout);
            const double temperature = monitor.temperature();
            ev::Settings settings;
            settings.repeats = cfg.repeats;
            settings.fixed_bits = task.bits;
            Warmup warmup(monitor, cfg.warmup);
            std::array<double, ev::Count> clock_sum{}, clock_weight{};
            settings.warmup = [&](auto name, const auto& issue) { warmup(name, issue); };
            settings.interval = [&](ev::Stage stage, double begin, double end) {
                const double clock = monitor.clock(begin, end);
                if (std::isfinite(clock)) {
                    clock_sum[stage] += clock * (end - begin);
                    clock_weight[stage] += end - begin;
                } else if (cfg.warmup > 0)
                    throw std::runtime_error("clock samples missing in a measured phase");
                warmup.measured(stage == ev::Compose ? "gemm" : ev::name(stage), end);
            };
            const double call_begin = ev::now();
            ttgemm::Info info;
            {
                ev::Scope scope(settings);
                info = context.gemm(a, b, c, task.n, task.n, task.n, task.options);
            }
            const double call_ms = (ev::now() - call_begin) * 1000;
            const std::uint64_t hash = paper_eval::output_hash(c);  // written in every mode
            const auto accuracy = cfg.reference ? paper_eval::accuracy(c, hi, lo) : paper_eval::Acc{NAN, NAN};
            if (cfg.reference && !std::isfinite(accuracy.rel_fro))
                throw std::runtime_error("nonfinite accuracy result");
            const double phase_sum = std::accumulate(settings.seconds.begin(), settings.seconds.end(), 0.0);
            if (!(phase_sum > 0) || !settings.partial_products)
                throw std::runtime_error("phase timing was not recorded");
            // The paper's total is the uninterrupted whole pass; the phase sum is the breakdown.
            if (!settings.whole_checked) {
                std::ostringstream why;
                why << "whole-pass replay check failed (steps run/expected:";
                for (int k = 0; k < ev::StepCount; ++k)
                    why << ' ' << settings.whole_ran[k] << '/' << settings.whole_expected[k];
                why << ")";
                throw std::runtime_error(why.str());
            }
            const double total = settings.whole_seconds;
            if (!(total > 0)) throw std::runtime_error("whole-pass timing was not recorded");
            const double operations = 2.0 * task.n * task.n * task.n;
            const double tops = operations * settings.partial_products / settings.seconds[ev::Gemm] / 1e12;
            const double peak =
                4096.0 * 1.35e9 * settings.cores / (task.options.format == ttgemm::Format::INT8 ? 4 : 1) / 1e12;
            const char* equivalent = !cfg.reference               ? "unknown"
                                     : accuracy.ebits < 53        ? "no"
                                     : !std::isfinite(cpu_bits)   ? "unknown"
                                     : accuracy.ebits >= cpu_bits ? "yes"
                                                                  : "no";
            csv << task.suite << ',' << task.n << ',' << scheme(task.options.scheme) << ','
                << ttgemm::format_name(task.options.format) << ',' << task.phi << ','
                << (cfg.warmup > 0 ? "steady" : "quick") << ',' << (cfg.reference ? "Dot2" : "none") << ','
                << info.slices << ',' << info.integer_bits << ',' << info.moduli << ',' << info.reconstruction_passes
                << ',' << settings.partial_products << ',' << settings.bandwidth;
            for (double seconds : settings.seconds) csv << ',' << seconds * 1000;
            csv << ',' << phase_sum * 1000 << ',' << total * 1000 << ',' << operations / total / 1e12 << ',' << tops
                << ',' << 100 * tops / peak << ','
                << (clock_weight[ev::Gemm] > 0 ? clock_sum[ev::Gemm] / clock_weight[ev::Gemm] : NAN) << ',' << call_ms
                << ',' << accuracy.ebits << ',' << accuracy.rel_fro << ',' << std::hex << hash << std::dec << ','
                << cpu_bits << ',' << equivalent << ',' << temperature << '\n';
            csv.flush();
            std::cout << "  " << operations / total / 1e12 << " TFlops/s (" << total * 1000
                      << " ms whole pass, phase sum " << phase_sum * 1000 << " ms), " << accuracy.ebits
                      << " bits, FP64-equivalent=" << equivalent << std::endl;
        }
        std::cout << "CSV: " << std::filesystem::absolute(path) << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Evaluation failed: " << e.what() << '\n';
        return 1;
    }
}
