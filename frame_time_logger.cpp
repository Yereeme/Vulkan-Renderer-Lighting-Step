#include "Frame_Time_Logger.hpp"
#include <cstdio>
#include <chrono>

using clock_type = std::chrono::high_resolution_clock;

FrameTimeLogger::FrameTimeLogger(bool headless_, const char* out_path) {
    headless = headless_;
    frame_count = 0;

    start_time = clock_type::now();
    last_time = start_time;

    file = nullptr;

    if (headless) {
        if (fopen_s(&file, out_path, "w") != 0) {
            file = nullptr;
        }

        if (file) {
            std::fprintf(file, "t_ms,frame_ms,fps\n");
            std::fflush(file);
        }
    }
}

FrameTimeLogger::~FrameTimeLogger() {
    if (file) {
        std::fflush(file);
        std::fclose(file);
        file = nullptr;
    }
}

void FrameTimeLogger::tick(double /*unused*/) {

    auto now = clock_type::now();

    double t_ms =
        std::chrono::duration<double, std::milli>(now - start_time).count();

    double frame_ms =
        std::chrono::duration<double, std::milli>(now - last_time).count();

    last_time = now;

    double fps = (frame_ms > 0.0) ? (1000.0 / frame_ms) : 0.0;

    if (headless) {
        if (file) {
            std::fprintf(file, "%.3f,%.6f,%.3f\n", t_ms, frame_ms, fps);
            std::fflush(file);
        }
        return;
    }

    // interactive mode: print once per second
    frame_count++;

    static auto interactive_last = clock_type::now();
    auto interactive_now = clock_type::now();

    std::chrono::duration<double> dt =
        interactive_now - interactive_last;

    if (dt.count() >= 1.0) {
        std::printf("\rFPS: %6.1f", fps);
        std::fflush(stdout);

        frame_count = 0;
        interactive_last = interactive_now;
    }
}