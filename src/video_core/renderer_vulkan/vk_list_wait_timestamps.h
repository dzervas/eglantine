// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class Scheduler;

/// EDEN_NX2_GPU_PROFILE_LIST_WAITS=1 (with EDEN_NX2_GPU_PROFILE): splits each wait of the front
/// for a GPU-written command list's copy (BufferCache::Nx2ReadListCopy) into the time until the
/// copy itself completed on the GPU, until the rest of its submission completed, and until the
/// waiting thread woke (Nx2Profile list wait counters). Timestamp queries after the copies and at
/// the end of their submission, put on the host's CLOCK_MONOTONIC (steady_clock) with
/// VK_EXT_calibrated_timestamps.
class ListWaitTimestamps {
public:
    static constexpr u64 NoStamp = ~u64{0};

    /// Null when not asked for or not possible (no calibrated timestamps, or Windows).
    static std::unique_ptr<ListWaitTimestamps> Create(const Device& device, Scheduler& scheduler);

    explicit ListWaitTimestamps(const Device& device, Scheduler& scheduler);

    /// Back thread, before the first list copy recorded for `tick`: the queries its stamp uses.
    u64 Reserve(u64 tick);
    /// Back thread, after the copies: records the copy timestamp, and the submission's.
    void Write(u64 stamp);

    /// Front: `wait` waited for the copies of `stamp`'s tick from `start_ns` to `end_ns`
    /// (steady_clock); `submitted` whether the tick was submitted when it began.
    void Measure(u64 stamp, u64 start_ns, u64 end_ns, bool submitted);

    [[nodiscard]] static u64 HostNow();

private:
    static constexpr u32 QueryCount = 16384;

    u32 NextQuery();
    bool ReadQuery(u32 query, u64& host_ns);
    bool Calibrate();

    const Device& device;
    Scheduler& scheduler;
    vk::QueryPool pool;
    double period_ns{};

    // Back thread
    u32 next_query{};
    u64 stamped_tick{~u64{0}};
    u32 end_query{};
    bool end_written{};

    // Front thread
    u64 calibrated_at_ns{};
    u64 calibration_gpu{};
    u64 calibration_host_ns{};
};

} // namespace Vulkan
