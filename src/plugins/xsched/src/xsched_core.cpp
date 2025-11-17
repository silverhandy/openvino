// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "xsched_core.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <type_traits>

#include "openvino/core/except.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/util/common_util.hpp"
#include "plugin.hpp"
#include "sync_infer_request.hpp"

namespace ov {
namespace xsched_plugin {
namespace {

constexpr double kEpsilon = 1e-6;

ov::ProfilingInfo make_runtime_profiling(const std::string& name, std::chrono::steady_clock::duration duration) {
    ov::ProfilingInfo info;
    info.status = ov::ProfilingInfo::Status::EXECUTED;
    info.node_name = name;
    info.cpu_time = info.real_time = std::chrono::duration_cast<std::chrono::microseconds>(duration);
    return info;
}

ov::Any parse_value(const std::string& value) {
    const auto trimmed = ov::util::trim(value);
    if (trimmed.empty()) {
        return value;
    }
    if (trimmed == "true" || trimmed == "TRUE" || trimmed == "True") {
        return true;
    }
    if (trimmed == "false" || trimmed == "FALSE" || trimmed == "False") {
        return false;
    }
    char* end_ptr = nullptr;
    long long integer = std::strtoll(trimmed.c_str(), &end_ptr, 10);
    if (end_ptr && *end_ptr == '\0') {
        return static_cast<int64_t>(integer);
    }
    if (trimmed.find('.') != std::string::npos || trimmed.find('e') != std::string::npos ||
        trimmed.find('E') != std::string::npos) {
        char* float_end = nullptr;
        double real = std::strtod(trimmed.c_str(), &float_end);
        if (float_end && *float_end == '\0') {
            return real;
        }
    }
    return value;
}

double parse_double(const std::map<std::string, std::string>& params,
                     const std::string& key,
                     double fallback) {
    auto it = params.find(key);
    if (it == params.end()) {
        return fallback;
    }
    try {
        return std::stod(it->second);
    } catch (...) {
        return fallback;
    }
}

}  // namespace

XschedCore::SoftwareHwQueue::SoftwareHwQueue(HwQueueHandle handle, XDevice device)
    : handle_(handle), device_(device) {}

XschedCore::SoftwareHwQueue::~SoftwareHwQueue() = default;

ov::InferRequest XschedCore::DeviceRuntime::acquire_request() {
    std::lock_guard<std::mutex> lock(request_pool_mutex);
    if (!request_pool.empty()) {
        auto request = std::move(request_pool.back());
        request_pool.pop_back();
        return request;
    }
    return compiled_model->create_infer_request();
}

void XschedCore::DeviceRuntime::release_request(ov::InferRequest&& request) {
    if (!request) {
        return;
    }
    std::lock_guard<std::mutex> lock(request_pool_mutex);
    request_pool.emplace_back(std::move(request));
}

void XschedCore::SoftwareHwQueue::Launch(std::shared_ptr<xsched::preempt::HwCommand> hw_cmd) {
    auto callback = std::dynamic_pointer_cast<xsched::preempt::HwCallbackCommand>(hw_cmd);
    if (callback) {
        callback->Launch(handle_);
    } else {
        // Non-callback commands are not expected in the software queue.
        hw_cmd->SetState(xsched::preempt::kCommandStateCompleted);
    }
}

void XschedCore::SoftwareHwQueue::Synchronize() {
    // Commands are executed synchronously inside the callback, so nothing to do here.
}

XschedCore::~XschedCore() {
    release_resources();
}

std::shared_ptr<XschedCore> XschedCore::create(const Plugin& plugin,
                                               const std::shared_ptr<const ov::Model>& model,
                                               const Configuration& cfg) {
    if (!model) {
        OPENVINO_THROW("XSCHED plugin received empty model");
    }

    std::vector<DeviceGroup> groups;
    groups.reserve(cfg.device_priority_groups.size());

    auto core = plugin.get_core();
    OPENVINO_ASSERT(core, "OpenVINO core is not initialized");

    std::random_device rd;

    auto make_group = [&](const Configuration::DeviceGroupConfig& group_cfg) {
        DeviceGroup group;
        group.label = group_cfg.label.empty() ? std::string{"group"} : group_cfg.label;
        group.params.type = parse_strategy(group_cfg.strategy);
        group.params.smoothing = parse_double(group_cfg.parameters, "smoothing", group.params.smoothing);
        group.params.min_weight = parse_double(group_cfg.parameters, "min_weight", group.params.min_weight);
        group.params.max_weight = parse_double(group_cfg.parameters, "max_weight", group.params.max_weight);
        if (group.params.smoothing <= 0.0 || group.params.smoothing >= 1.0) {
            group.params.smoothing = std::clamp(group.params.smoothing, 0.01, 0.9);
        }
        if (group.params.min_weight <= 0.0) {
            group.params.min_weight = 0.01;
        }
        if (group.params.max_weight < group.params.min_weight) {
            group.params.max_weight = std::max(group.params.min_weight, 1.0);
        }
        group.rng.seed(rd());

        for (const auto& device_cfg : group_cfg.devices) {
            if (device_cfg.id.empty()) {
                continue;
            }

            ov::AnyMap device_properties;
            for (const auto& [key, raw_value] : device_cfg.parameters) {
                device_properties.emplace(key, parse_value(raw_value));
            }

            auto compiled = core->compile_model(model, device_cfg.id, device_properties);

            auto runtime = std::make_shared<DeviceRuntime>();
            runtime->name = device_cfg.id;
            runtime->compiled_model = std::move(compiled);
            runtime->xsched_device = make_xdevice(device_cfg.id);

            auto hw_handle = make_queue_handle();
            auto add_res =
                xsched::preempt::HwQueueManager::Add(hw_handle, [&]() {
                    return std::make_shared<SoftwareHwQueue>(hw_handle, runtime->xsched_device);
                });
            if (add_res != kXSchedSuccess) {
                OPENVINO_THROW("Failed to register HwQueue for device ", device_cfg.id, ", error ", add_res);
            }

            XQueueHandle xqueue = 0;
            auto res = XQueueCreate(&xqueue,
                                    hw_handle,
                                    static_cast<int64_t>(kPreemptLevelBlock),
                                    kQueueCreateFlagNone);
            if (res != kXSchedSuccess) {
                HwQueueDestroy(hw_handle);
                OPENVINO_THROW("Failed to create XQueue for device ", device_cfg.id, ", error ", res);
            }

            runtime->hw_queue = hw_handle;
            runtime->x_queue = xqueue;
            runtime->moving_avg.store(0.0);
            runtime->weight.store(1.0);

            group.devices.emplace_back(std::move(runtime));
        }

        if (!group.devices.empty()) {
            groups.emplace_back(std::move(group));
        }
    };

    if (!cfg.device_priority_groups.empty()) {
        for (const auto& group_cfg : cfg.device_priority_groups) {
            make_group(group_cfg);
        }
    }

    if (groups.empty()) {
        // Fallback: single group using default ov::device::priorities string list
        Configuration::DeviceGroupConfig default_cfg;
        default_cfg.label = "default";
        auto tokens = ov::util::split(cfg.device_priorities_raw, ',', true);
        for (auto& token : tokens) {
            Configuration::DeviceConfig device;
            device.id = token;
            default_cfg.devices.push_back(std::move(device));
        }
        if (default_cfg.devices.empty()) {
            Configuration::DeviceConfig cpu;
            cpu.id = "CPU";
            default_cfg.devices.push_back(std::move(cpu));
        }
        make_group(default_cfg);
    }

    if (groups.empty()) {
        OPENVINO_THROW("XSCHED plugin failed to create any device group");
    }

    return std::shared_ptr<XschedCore>(new XschedCore(model, cfg, std::move(groups)));
}

std::vector<std::string> XschedCore::get_device_names() const {
    std::vector<std::string> names;
    for (const auto& group : m_groups) {
        for (const auto& device : group.devices) {
            if (device) {
                names.push_back(device->name);
            }
        }
    }
    return names;
}

XschedCore::XschedCore(const std::shared_ptr<const ov::Model>& model,
                       const Configuration& cfg,
                       std::vector<DeviceGroup>&& groups)
    : m_model(model), m_cfg(cfg), m_groups(std::move(groups)) {}

std::shared_ptr<XschedPendingCommand> XschedCore::enqueue(InferRequest& request) {
    if (m_groups.empty()) {
        OPENVINO_THROW("XSCHED runtime is not initialized");
    }

    // Select the best group based on current load.
    DeviceGroup* chosen_group = nullptr;
    double best_score = std::numeric_limits<double>::max();
    for (auto& group : m_groups) {
        if (group.devices.empty()) {
            continue;
        }
        double load_sum = 0.0;
        for (const auto& device : group.devices) {
            double avg = device->moving_avg.load();
            double inflight = static_cast<double>(device->inflight.load());
            load_sum += avg + inflight;
        }
        if (load_sum < best_score) {
            best_score = load_sum;
            chosen_group = &group;
        }
    }

    if (!chosen_group) {
        OPENVINO_THROW("XSCHED runtime found no available device group");
    }

    std::shared_ptr<DeviceRuntime> device;
    {
        std::lock_guard<std::mutex> lock(chosen_group->mutex);
        device = select_device(*chosen_group);
    }

    if (!device) {
        OPENVINO_THROW("XSCHED runtime could not choose device for inference");
    }

    auto pending = std::make_shared<XschedPendingCommand>();
    pending->runtime = this;
    pending->group = chosen_group;
    pending->device = device;
    pending->request = &request;
    pending->enable_profiling = m_enable_profiling.load();
    pending->queued_at = std::chrono::steady_clock::now();
    pending->future = pending->completion.get_future();

    HwCommandHandle hw_command = 0;
    auto result = HwCommandCreateCallback(&hw_command, &XschedCore::launch_callback, pending.get());
    if (result != kXSchedSuccess) {
        OPENVINO_THROW("XSCHED runtime failed to create HwCommand, error ", result);
    }

    pending->command = hw_command;

    {
        std::lock_guard<std::mutex> lock(m_pending_mutex);
        m_pending.emplace(hw_command, pending);
    }

    pending->device->inflight.fetch_add(1, std::memory_order_relaxed);
    pending->device->submitted.fetch_add(1, std::memory_order_relaxed);

    auto submit_result = XQueueSubmit(device->x_queue, hw_command);
    if (submit_result != kXSchedSuccess) {
        pending->device->inflight.fetch_sub(1, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(m_pending_mutex);
            m_pending.erase(hw_command);
        }
        HwCommandDestroy(hw_command);
        OPENVINO_THROW("XSCHED runtime failed to submit command, error ", submit_result);
    }

    return pending;
}

XschedCore::StrategyType XschedCore::parse_strategy(const std::string& value) {
    auto upper = ov::util::to_upper(ov::util::trim(value));
    if (upper == "ROUNDROBIN" || upper == "RR" || upper.empty()) {
        return StrategyType::RoundRobin;
    }
    if (upper == "LOADBALANCE" || upper == "LB" || upper == "LOAD-BALANCE" || upper == "LEASTLOADED" ||
        upper == "LEAST-LOADED") {
        return StrategyType::LeastLoaded;
    }
    if (upper == "GA" || upper == "GENETIC" || upper == "DE" || upper == "DIFFERENTIAL" ||
        upper == "PSO" || upper == "PARTICLE" || upper == "ADAPTIVE" || upper == "WEIGHTED") {
        return StrategyType::Adaptive;
    }
    return StrategyType::RoundRobin;
}

XDevice XschedCore::make_xdevice(const std::string& name) {
    auto trimmed = ov::util::trim(name);
    auto upper = ov::util::to_upper(trimmed);
    XDeviceType type = kDeviceTypeUnknown;
    if (upper.find("GPU") == 0) {
        type = kDeviceTypeGPU;
    } else if (upper.find("NPU") == 0) {
        type = kDeviceTypeNPU;
    } else {
        type = kDeviceTypeCPU;
    }

    std::string::size_type dot_pos = trimmed.find('.');
    uint32_t index = 0;
    if (dot_pos != std::string::npos && dot_pos + 1 < trimmed.size()) {
        try {
            index = static_cast<uint32_t>(std::stoul(trimmed.substr(dot_pos + 1)));
        } catch (...) {
            index = 0;
        }
    }

    return xsched::protocol::MakeDevice(type, static_cast<XDeviceId>(index));
}

HwQueueHandle XschedCore::make_queue_handle() {
    static std::atomic<uint64_t> counter{1};
    return counter.fetch_add(1, std::memory_order_relaxed);
}

std::shared_ptr<XschedCore::DeviceRuntime> XschedCore::select_device(DeviceGroup& group) {
    switch (group.params.type) {
    case StrategyType::RoundRobin:
        return select_round_robin(group);
    case StrategyType::LeastLoaded:
        return select_least_loaded(group);
    case StrategyType::Adaptive:
        return select_adaptive(group);
    default:
        return select_round_robin(group);
    }
}

std::shared_ptr<XschedCore::DeviceRuntime> XschedCore::select_round_robin(DeviceGroup& group) {
    const auto size = group.devices.size();
    if (size == 0) {
        return nullptr;
    }
    auto index = group.round_robin_cursor.fetch_add(1, std::memory_order_relaxed) % size;
    return group.devices[index];
}

std::shared_ptr<XschedCore::DeviceRuntime> XschedCore::select_least_loaded(DeviceGroup& group) {
    std::shared_ptr<DeviceRuntime> best;
    double best_score = std::numeric_limits<double>::max();
    for (auto& candidate : group.devices) {
        const double inflight = static_cast<double>(candidate->inflight.load(std::memory_order_relaxed));
        const double avg = candidate->moving_avg.load(std::memory_order_relaxed);
        const double score = inflight * (avg + 1.0) + avg;
        if (!best || score < best_score) {
            best = candidate;
            best_score = score;
        }
    }
    return best;
}

std::shared_ptr<XschedCore::DeviceRuntime> XschedCore::select_adaptive(DeviceGroup& group) {
    if (group.devices.empty()) {
        return nullptr;
    }

    double total = 0.0;
    for (const auto& device : group.devices) {
        double weight = device->weight.load(std::memory_order_relaxed);
        if (!std::isfinite(weight) || weight <= 0.0) {
            weight = 1.0;
        }
        total += weight;
    }

    if (total <= 0.0) {
        return select_least_loaded(group);
    }

    std::uniform_real_distribution<double> dist(0.0, total);
    double threshold = dist(group.rng);
    double cumulative = 0.0;
    for (auto& device : group.devices) {
        double weight = device->weight.load(std::memory_order_relaxed);
        if (!std::isfinite(weight) || weight <= 0.0) {
            weight = 1.0;
        }
        cumulative += weight;
        if (threshold <= cumulative) {
            return device;
        }
    }

    return group.devices.back();
}

void XschedCore::update_strategy(DeviceGroup& group, DeviceRuntime& device, double latency_ms) {
    const double smoothing = group.params.smoothing;
    double previous_avg = device.moving_avg.load(std::memory_order_relaxed);
    double updated_avg = latency_ms;
    if (previous_avg > 0.0) {
        updated_avg = (1.0 - smoothing) * previous_avg + smoothing * latency_ms;
    }
    device.moving_avg.store(updated_avg, std::memory_order_relaxed);

    double weight = 1.0 / std::max(updated_avg, kEpsilon);
    weight = std::clamp(weight, group.params.min_weight, group.params.max_weight);
    device.weight.store(weight, std::memory_order_relaxed);
}

XResult XschedCore::launch_callback(HwQueueHandle /*hwq*/, void* data) noexcept {
    auto* pending = reinterpret_cast<XschedPendingCommand*>(data);
    if (!pending || !pending->runtime) {
        return kXSchedErrorInvalidValue;
    }
    pending->runtime->execute_pending(*pending);
    return kXSchedSuccess;
}

void XschedCore::execute_pending(XschedPendingCommand& pending) noexcept {
    pending.started_at = std::chrono::steady_clock::now();
    try {
        pending.request->execute_on_device(pending);
    } catch (...) {
        pending.error = std::current_exception();
    }
    pending.finished_at = std::chrono::steady_clock::now();

    const double latency_ms = std::chrono::duration_cast<std::chrono::microseconds>(pending.finished_at -
                                                                                    pending.started_at)
                                  .count() /
                              1000.0;

    if (pending.enable_profiling) {
        try {
            auto total_duration = pending.finished_at - pending.queued_at;
            auto exec_duration = pending.finished_at - pending.started_at;
            auto wait_duration = pending.started_at - pending.queued_at;
            pending.profiling.insert(pending.profiling.begin(),
                                     make_runtime_profiling("xsched.total_latency", total_duration));
            pending.profiling.insert(pending.profiling.begin(),
                                     make_runtime_profiling("xsched.device_execution", exec_duration));
            pending.profiling.insert(pending.profiling.begin(),
                                     make_runtime_profiling("xsched.queue_wait", wait_duration));
        } catch (...) {
        }
    }

    if (pending.group && pending.device) {
        std::lock_guard<std::mutex> lock(pending.group->mutex);
        update_strategy(*pending.group, *pending.device, latency_ms);
    }

    pending.device->inflight.fetch_sub(1, std::memory_order_relaxed);
    pending.device->completed.fetch_add(1, std::memory_order_relaxed);
    pending.device->last_latency.store(latency_ms, std::memory_order_relaxed);

    if (pending.error) {
        try {
            pending.completion.set_exception(pending.error);
        } catch (...) {
        }
    } else {
        try {
            pending.completion.set_value();
        } catch (...) {
        }
    }

    {
        std::lock_guard<std::mutex> lock(m_pending_mutex);
        m_pending.erase(pending.command);
    }

    if (pending.command != 0) {
        HwCommandDestroy(pending.command);
        pending.command = 0;
    }
}

void XschedCore::release_resources() {
    for (auto& group : m_groups) {
        for (auto& device : group.devices) {
            if (device->x_queue != 0) {
                XQueueDestroy(device->x_queue);
                device->x_queue = 0;
            }
            if (device->hw_queue != 0) {
                HwQueueDestroy(device->hw_queue);
                xsched::preempt::HwQueueManager::Del(device->hw_queue);
                device->hw_queue = 0;
            }
        }
    }
}

}  // namespace xsched_plugin
}  // namespace ov
