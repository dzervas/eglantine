// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <memory>

#include "common/logging.h"
#include "video_core/nx2_profile.h"
#include "video_core/renderer_vulkan/vk_list_wait_timestamps.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {

namespace Profile = VideoCommon::Nx2Profile;

std::unique_ptr<ListWaitTimestamps> ListWaitTimestamps::Create(const Device& device,
                                                               Scheduler& scheduler) {
    if (!Profile::Enabled() || std::getenv("EDEN_NX2_GPU_PROFILE_LIST_WAITS") == nullptr) {
        return nullptr;
    }
#ifdef _WIN32
    // steady_clock is QueryPerformanceCounter there, in its own units: not calibrated here.
    LOG_WARNING(Render_Vulkan, "EDEN_NX2_GPU_PROFILE_LIST_WAITS: not supported on Windows");
    return nullptr;
#else
    if (!device.HasCalibratedTimestamps()) {
        LOG_WARNING(Render_Vulkan,
                    "EDEN_NX2_GPU_PROFILE_LIST_WAITS: no VK_EXT_calibrated_timestamps");
        return nullptr;
    }
    auto timestamps = std::make_unique<ListWaitTimestamps>(device, scheduler);
    if (!timestamps->Calibrate()) {
        LOG_WARNING(Render_Vulkan,
                    "EDEN_NX2_GPU_PROFILE_LIST_WAITS: the device and CLOCK_MONOTONIC can't be "
                    "calibrated");
        return nullptr;
    }
    LOG_INFO(Render_Vulkan, "EDEN_NX2_GPU_PROFILE_LIST_WAITS: list waits are split");
    return timestamps;
#endif
}

ListWaitTimestamps::ListWaitTimestamps(const Device& device_, Scheduler& scheduler_)
    : device{device_}, scheduler{scheduler_},
      pool{device.GetLogical().CreateQueryPool({
          .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
          .pNext = nullptr,
          .flags = 0,
          .queryType = VK_QUERY_TYPE_TIMESTAMP,
          .queryCount = QueryCount,
          .pipelineStatistics = 0,
      })},
      period_ns{device.GetTimestampPeriod()} {}

u64 ListWaitTimestamps::HostNow() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

u32 ListWaitTimestamps::NextQuery() {
    const u32 query = next_query;
    next_query = (next_query + 1) % QueryCount;
    return query;
}

u64 ListWaitTimestamps::Reserve(u64 tick) {
    if (tick != stamped_tick) {
        stamped_tick = tick;
        end_query = NextQuery();
        end_written = false;
    }
    return (u64{end_query} << 32) | NextQuery();
}

void ListWaitTimestamps::Write(u64 stamp) {
    const u32 copy_query = static_cast<u32>(stamp);
    const bool first_of_tick = !end_written;
    end_written = true;
    // Resets go outside a render pass; the copies before are already there.
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([pool = *pool, copy_query, end = static_cast<u32>(stamp >> 32),
                      first_of_tick](vk::CommandBuffer cmdbuf) {
        if (first_of_tick) {
            cmdbuf.ResetQueryPool(pool, end, 1);
        }
        cmdbuf.ResetQueryPool(pool, copy_query, 1);
        cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_TRANSFER_BIT, pool, copy_query);
    });
    if (first_of_tick) {
        scheduler.WriteTimestampAtSubmit(*pool, static_cast<u32>(stamp >> 32));
    }
}

bool ListWaitTimestamps::Calibrate() {
    const std::array<VkCalibratedTimestampInfoEXT, 2> infos{{
        {VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT, nullptr, VK_TIME_DOMAIN_DEVICE_EXT},
        {VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT, nullptr,
         VK_TIME_DOMAIN_CLOCK_MONOTONIC_EXT},
    }};
    std::array<u64, 2> values{};
    u64 deviation{};
    if (device.GetLogical().GetCalibratedTimestamps(infos, values.data(), &deviation) !=
        VK_SUCCESS) {
        return false;
    }
    calibration_gpu = values[0];
    calibration_host_ns = values[1];
    calibrated_at_ns = HostNow();
    return true;
}

bool ListWaitTimestamps::ReadQuery(u32 query, u64& host_ns) {
    u64 ticks{};
    if (device.GetLogical().GetQueryResults(*pool, query, 1, sizeof(ticks), &ticks, sizeof(ticks),
                                            VK_QUERY_RESULT_64_BIT) != VK_SUCCESS) {
        return false;
    }
    const double delta = static_cast<double>(static_cast<s64>(ticks - calibration_gpu)) * period_ns;
    host_ns = static_cast<u64>(static_cast<double>(calibration_host_ns) + delta);
    return true;
}

void ListWaitTimestamps::Measure(u64 stamp, u64 start_ns, u64 end_ns, bool submitted) {
    if (!submitted) {
        Profile::Add(Profile::Counter::ListWaitsUnsubmitted, 1);
    }
    // The clocks drift apart: calibrate again every second.
    if (end_ns - calibrated_at_ns > 1'000'000'000ULL && !Calibrate()) {
        Profile::Add(Profile::Counter::ListWaitsUnsplit, 1);
        return;
    }
    u64 copy_ns{};
    u64 submission_ns{};
    if (!ReadQuery(static_cast<u32>(stamp), copy_ns) ||
        !ReadQuery(static_cast<u32>(stamp >> 32), submission_ns)) {
        Profile::Add(Profile::Counter::ListWaitsUnsplit, 1);
        return;
    }
    // Each part from where the one before ended, within the wait (calibration is approximate).
    const u64 total = end_ns - start_ns;
    const auto since_start = [&](u64 ns) {
        return ns <= start_ns ? u64{0} : (std::min)(ns - start_ns, total);
    };
    const u64 until_copy = since_start(copy_ns);
    const u64 until_submission = (std::max)(since_start(submission_ns), until_copy);
    Profile::Add(Profile::Counter::ListWaitsSplit, 1);
    Profile::Add(Profile::Counter::ListWaitUntilCopyNs, until_copy);
    Profile::Add(Profile::Counter::ListWaitUntilSubmitNs, until_submission - until_copy);
    Profile::Add(Profile::Counter::ListWaitWakeNs, total - until_submission);
}

} // namespace Vulkan
