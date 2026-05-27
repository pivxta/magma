#pragma once
#include <cstdint>
#include <vector>
#include <optional>
#include <algorithm>
#include <numeric>
#include "time.h"

struct PerformanceStats {
    double frames_per_second;
    double frame_time_high_1pct_ms;
    double frame_time_high_0_1pct_ms;
    double frame_time_median_ms;
    double frame_time_avg_ms;
};

class PerformanceCounter {
public:
    PerformanceCounter(): 
        last_frame(Instant::now()),
        last_clean(Instant::now())
    {
        this->frame_times.reserve(2048);
    } 

    void record() {
        this->frame_times.push_back(this->last_frame.elapsed_seconds<float>());
        this->last_frame = Instant::now();
    }

    std::optional<PerformanceStats> record(double stat_interval_secs) {
        this->record();
        if (this->last_clean.elapsed_seconds<double>() >= stat_interval_secs) {
            PerformanceStats stats = this->get_stats();
            this->last_clean = Instant::now();
            this->frame_times.clear();
            return stats;
        }
        return std::nullopt;
    }

    PerformanceStats get_stats() {
        if (this->frame_times.empty()) {
            auto since_last_clean_ms = this->last_clean.elapsed_milliseconds<double>();
            return {
                .frames_per_second = 0.0,
                .frame_time_high_1pct_ms = since_last_clean_ms,
                .frame_time_high_0_1pct_ms = since_last_clean_ms,
                .frame_time_median_ms = since_last_clean_ms,
                .frame_time_avg_ms = since_last_clean_ms,
            };    
        }

        std::sort(this->frame_times.begin(), this->frame_times.end());

        double avg = 
            std::accumulate(this->frame_times.begin(), this->frame_times.end(), 0.0) 
                / static_cast<double>(this->frame_times.size());
        double median = this->frame_times[this->frame_times.size() / 2];

        size_t one_pct = std::max(static_cast<size_t>(1), this->frame_times.size() / 100);
        size_t point_one_pct = std::max(static_cast<size_t>(1), this->frame_times.size() / 1000);
        double low_1pct = average_tail(this->frame_times, one_pct);
        double low_0_1pct = average_tail(this->frame_times, point_one_pct);

        auto since_last_clean = this->last_clean.elapsed_seconds<double>();
        return {
            .frames_per_second = static_cast<double>(this->frame_times.size()) / since_last_clean,
            .frame_time_high_1pct_ms = low_1pct * 1000.0,
            .frame_time_high_0_1pct_ms = low_0_1pct * 1000.0,
            .frame_time_median_ms = median * 1000.0,
            .frame_time_avg_ms = avg * 1000.0,
        };
    }

private:
    static double average_tail(const std::vector<float>& sorted, size_t count) {
        double sum = 0.0;
        for (size_t i = sorted.size() - count; i < sorted.size(); i++) {
            sum += sorted[i];
        }
        return sum / static_cast<double>(count);
    }

    std::vector<float> frame_times;
    Instant last_frame;
    Instant last_clean;
};