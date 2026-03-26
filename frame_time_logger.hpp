#pragma once
 
#include <cstdio>
#include <string>

#include <chrono>

class FrameTimeLogger {
public:
    FrameTimeLogger(bool headless, const char* out_path);
    ~FrameTimeLogger();

    void tick(double frame_ms);

private:
    using clock = std::chrono::high_resolution_clock;

    bool headless;
    FILE* file;

    std::chrono::time_point<clock> start_time;
    std::chrono::time_point<clock> last_time;

    int frame_count;
};