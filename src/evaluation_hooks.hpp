// SPDX-License-Identifier: Apache-2.0
#pragma once
// Private hooks for the optional paper evaluator. Ordinary GEMM executes each
// operation once; only an explicit Scope enables warmup, repetition and timing.
//
// Two timings are taken inside one profiled call:
//  * phase timing: each stage is issued once (JIT), warmed, then repeated and
//    synchronized; the per-phase means give the phase breakdown;
//  * whole-pass timing: after the last stage, the complete sequence of device steps
//    is issued again without any synchronization between stages; the buffers normally
//    released between stages are kept by the backend (its may_release). Scheme II
//    records every step it issues during the phase timing (record) and replays the
//    records; scheme I issues its stage loops again inline, rebuilding the panel
//    programs first when there is a single K block. The pass is checked first
//    (sentinel fill, bit-identical output, step counts) and then warmed and timed.
//    This whole-pass time is the paper's "Total"; the phase sum is the breakdown only.
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace ozTTgemm::detail::evaluation {
enum Stage { Scale, Slice, Gemm, Compose, Recon, Count };
inline const char* name(Stage stage) {
    constexpr const char* names[]{"scale", "slice", "gemm", "compose", "recon"};
    return names[stage];
}
// Kinds of recorded device steps: the scaling and slicing programs (Head), one partial-product
// GEMM, one reconstruction program, and one composition program (Ozaki II).
enum Step { Head, GemmStep, ReconStep, ComposeStep, StepCount };
struct Counts {
    std::array<unsigned long long, StepCount> n{};
    unsigned long long& operator[](int k) { return n[k]; }
    unsigned long long operator[](int k) const { return n[k]; }
    bool operator==(const Counts& o) const { return n == o.n; }
};
// Sentinel written to every intermediate and output buffer before the replay check:
// NaN in FP32/BF16, a huge finite value in FP64, so an unwritten buffer changes the output.
constexpr std::uint32_t SENTINEL = 0x7FC07FC0u;
inline double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
// One recorded device step of the whole pass: its kind and the closure that issues it.
struct RecordedStep {
    Step kind;
    std::function<void()> issue;
};
// Inputs of a profiled call and its results. The results are reset when a Scope
// starts, so one Settings can serve several Scopes in turn.
struct Settings {
    // Inputs.
    unsigned repeats = 3;
    unsigned fixed_bits = 0;  // zero: normal input-dependent Ozaki II selection
    std::function<void(std::string_view, const std::function<void()>&)> warmup;
    std::function<void(Stage, double, double)> interval;
    bool whole = true;  // whole-pass timing (on by default inside a Scope)
    // Results.
    unsigned partial_products = 0, bandwidth = 0, cores = 0;
    std::array<double, Count> seconds{};  // mean time per stage
    bool recording = false;               // set by sweep while its timed repetitions run
    double whole_seconds = NAN;           // mean of one uninterrupted pass over all device steps
    bool whole_checked = false;           // sentinel, replay-identical and step-count checks all passed
    Counts whole_expected, whole_ran;
    // Device steps recorded for the whole pass. They borrow state of the call that
    // recorded them and live from the start of that call (the Context discards earlier
    // records) to the end of its whole pass, at the latest to the end of the Scope.
    std::vector<RecordedStep> steps;
};
inline thread_local Settings* active = nullptr;
// Activates the settings for the GEMM calls made while the Scope exists (one Scope per
// call in the evaluator), after resetting their results.
class Scope {
public:
    explicit Scope(Settings& settings) : settings_(settings), previous_(active) {
        if (settings.repeats == 0) throw std::invalid_argument("evaluation repeats must be positive");
        settings.partial_products = settings.bandwidth = settings.cores = 0;
        settings.seconds.fill(0);
        settings.recording = false;
        settings.whole_seconds = NAN;
        settings.whole_checked = false;
        settings.whole_expected = settings.whole_ran = Counts{};
        settings.steps.clear();
        active = &settings;
    }
    ~Scope() {
        settings_.steps.clear();
        active = previous_;
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    Settings& settings_;
    Settings* previous_;
};
inline bool recording_whole() { return active && active->whole; }
// Record one device step of the whole pass (issue only, no synchronization). The
// closure runs after the recording function returned; it may borrow state of the
// current call (buffers, programs), because the recorded steps are discarded once the
// whole pass has been replayed (whole), when the next call starts and, at the latest,
// when the Scope ends.
inline void record(Step kind, std::function<void()> issue) {
    if (!recording_whole()) return;
    active->steps.push_back({kind, std::move(issue)});
}
inline void record(Stage stage, double begin, double end) {
    active->seconds[stage] += (end - begin) / active->repeats;
    if (active->interval) active->interval(stage, begin, end);
}
template <class F>
void phase(Stage stage, F&& issue) {
    if (!active || !active->recording) {
        issue();
        return;
    }
    const double begin = now();
    issue();
    record(stage, begin, now());
}
template <class F, class S>
void execute(Stage stage, F&& issue, S&& sync) {
    if (!active) {
        issue();
        sync();
        return;
    }
    issue();
    sync();  // JIT and initial execution are outside the measured interval.
    const auto group = stage == Compose ? "gemm" : name(stage);
    if (active->warmup)
        active->warmup(group, [&] {
            for (unsigned i = 0; i < active->repeats; ++i) issue();
            sync();
        });
    const double begin = now();
    for (unsigned i = 0; i < active->repeats; ++i) issue();
    sync();
    record(stage, begin, now());
}
template <class F>
void execute(Stage stage, F&& issue) {
    execute(stage, std::forward<F>(issue), [] {});
}
template <class F>
void sweep(F&& issue) {
    if (!active) {
        issue();
        return;
    }
    issue();
    if (active->warmup) active->warmup("sweep", issue);
    active->recording = true;
    try {
        for (unsigned i = 0; i < active->repeats; ++i) issue();
    } catch (...) {
        active->recording = false;
        throw;
    }
    active->recording = false;
}
// Check and time the whole pass. fill() writes SENTINEL into every buffer the pass
// writes (not the inputs), read_out(v) reads the whole output buffer, sync() waits for
// the queue, run_once(note) issues the complete pass calling note(kind) per step, and
// expected holds the step counts derived from the configuration. The recorded steps
// are dropped on every exit; the output buffer holds the replayed result, which must
// be bit-identical to the phase-timed one.
template <class Fill, class ReadOut, class Sync, class RunOnce>
void whole(Fill&& fill, ReadOut&& read_out, Sync&& sync, RunOnce&& run_once, const Counts& expected) {
    if (!recording_whole()) return;
    Settings& s = *active;
    struct Discard {
        std::vector<RecordedStep>& steps;
        ~Discard() { steps.clear(); }
    } discard{s.steps};
    std::vector<std::uint32_t> pre, got;
    read_out(pre);
    fill();
    read_out(got);
    bool filled = !got.empty();
    for (std::uint32_t v : got)
        if (v != SENTINEL) {
            filled = false;
            break;
        }
    Counts ran;
    run_once([&](int kind) { ++ran[kind]; });
    sync();
    read_out(got);
    const bool sentinel = filled && got == pre;
    auto pass = [&] { run_once([](int) {}); };
    if (s.warmup)
        s.warmup("whole", [&] {
            pass();
            sync();
        });
    const double begin = now();
    for (unsigned i = 0; i < s.repeats; ++i) pass();
    sync();
    s.whole_seconds = (now() - begin) / s.repeats;
    read_out(got);
    s.whole_expected = expected;
    s.whole_ran = ran;
    s.whole_checked = sentinel && ran == expected && got == pre;
}
// Issue the recorded steps in order (Ozaki II records each step; Ozaki I issues inline).
template <class Note>
void replay(Note&& note) {
    for (const RecordedStep& step : active->steps) {
        step.issue();
        note(step.kind);
    }
}
}  // namespace ozTTgemm::detail::evaluation
