// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <condition_variable>
#include <chrono>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "config.hpp"
#include "openvino/core/model.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "openvino/runtime/profiling_info.hpp"
#include "openvino/runtime/tensor.hpp"
#include "xsched/preempt/hal/hw_queue.h"
#include "xsched/protocol/device.h"
#include "xsched/types.h"
#include "xsched/xqueue.h"

namespace ov {
namespace xsched_plugin {

class Plugin;
class CompiledModel;
class InferRequest;
class XschedCore;
struct XschedPendingCommand;

class XschedCore : public std::enable_shared_from_this<XschedCore> {
public:
    enum class StrategyType {
        RoundRobin,
        LeastLoaded,
        Adaptive
    };

    ~XschedCore();

    static std::shared_ptr<XschedCore> create(const Plugin& plugin,
                                              const std::shared_ptr<const ov::Model>& model,
                                              const Configuration& cfg);

    std::shared_ptr<XschedPendingCommand> enqueue(InferRequest& request);

    void enable_profiling(bool enable) { m_enable_profiling = enable; }

    std::vector<std::string> get_device_names() const;

    struct StrategyParams {
        StrategyType type = StrategyType::RoundRobin;
        double smoothing = 0.2;
        double min_weight = 0.05;
        double max_weight = 10.0;
    };

    struct DeviceRuntime {
        std::string name;
        ov::SoPtr<ov::ICompiledModel> compiled_model;
        XDevice xsched_device{};
        HwQueueHandle hw_queue = 0;
        XQueueHandle x_queue = 0;
        std::atomic<uint64_t> inflight{0};
        std::atomic<uint64_t> submitted{0};
        std::atomic<uint64_t> completed{0};
        std::atomic<double> moving_avg{0.0};
        std::atomic<double> last_latency{0.0};
        std::atomic<double> weight{1.0};
    };

    struct DeviceGroup {
        DeviceGroup() = default;
        DeviceGroup(const DeviceGroup&) = delete;
        DeviceGroup& operator=(const DeviceGroup&) = delete;
        DeviceGroup(DeviceGroup&& other) noexcept {
            label = std::move(other.label);
            params = other.params;
            devices = std::move(other.devices);
            rng = std::move(other.rng);
            round_robin_cursor.store(other.round_robin_cursor.load(std::memory_order_relaxed),
                                     std::memory_order_relaxed);
            other.round_robin_cursor.store(0, std::memory_order_relaxed);
        }
        DeviceGroup& operator=(DeviceGroup&& other) noexcept {
            if (this != &other) {
                label = std::move(other.label);
                params = other.params;
                devices = std::move(other.devices);
                rng = std::move(other.rng);
                round_robin_cursor.store(other.round_robin_cursor.load(std::memory_order_relaxed),
                                         std::memory_order_relaxed);
                other.round_robin_cursor.store(0, std::memory_order_relaxed);
            }
            return *this;
        }

        std::string label;
        StrategyParams params;
        std::vector<std::shared_ptr<DeviceRuntime>> devices;
        std::mutex mutex;
        std::mt19937 rng;
        std::atomic<uint64_t> round_robin_cursor{0};
    };

private:
    class SoftwareHwQueue : public xsched::preempt::HwQueue {
    public:
        SoftwareHwQueue(HwQueueHandle handle, XDevice device);
        ~SoftwareHwQueue() override;

        void Launch(std::shared_ptr<xsched::preempt::HwCommand> hw_cmd) override;
        void Synchronize() override;
        XDevice GetDevice() override { return device_; }
        HwQueueHandle GetHandle() override { return handle_; }
        bool SupportDynamicLevel() override { return false; }
        XPreemptLevel GetMaxSupportedLevel() override {
            return kPreemptLevelBlock;
        }

    private:
        HwQueueHandle handle_;
        XDevice device_;
    };

    explicit XschedCore(const std::shared_ptr<const ov::Model>& model,
                        const Configuration& cfg,
                        std::vector<DeviceGroup>&& groups);

    std::shared_ptr<DeviceRuntime> select_device(DeviceGroup& group);
    std::shared_ptr<DeviceRuntime> select_round_robin(DeviceGroup& group);
    std::shared_ptr<DeviceRuntime> select_least_loaded(DeviceGroup& group);
    std::shared_ptr<DeviceRuntime> select_adaptive(DeviceGroup& group);

    void update_strategy(DeviceGroup& group, DeviceRuntime& device, double latency_ms);

    static StrategyType parse_strategy(const std::string& value);
    static XDevice make_xdevice(const std::string& name);
    static HwQueueHandle make_queue_handle();

    static XResult launch_callback(HwQueueHandle hwq, void* data) noexcept;
    void execute_pending(XschedPendingCommand& pending) noexcept;

    void release_resources();

    std::shared_ptr<const ov::Model> m_model;
    Configuration m_cfg;
    std::vector<DeviceGroup> m_groups;
    std::mutex m_pending_mutex;
    std::unordered_map<HwCommandHandle, std::shared_ptr<XschedPendingCommand>> m_pending;
    std::atomic<bool> m_enable_profiling{false};
};

struct XschedPendingCommand {
    XschedCore* runtime = nullptr;
    XschedCore::DeviceGroup* group = nullptr;
    std::shared_ptr<XschedCore::DeviceRuntime> device;
    InferRequest* request = nullptr;
    std::promise<void> completion;
    std::future<void> future;
    std::exception_ptr error;
    std::vector<ov::ProfilingInfo> profiling;
    bool enable_profiling = false;
    std::chrono::steady_clock::time_point queued_at;
    std::chrono::steady_clock::time_point started_at;
    std::chrono::steady_clock::time_point finished_at;
    HwCommandHandle command = 0;
};

}  // namespace xsched_plugin
}  // namespace ov
