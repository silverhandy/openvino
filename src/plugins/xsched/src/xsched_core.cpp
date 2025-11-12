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
namespace template_plugin {
namespace {

constexpr double kEpsilon = 1e-6;
constexpr double kAlpha = 0.2;  // moving average smoothing factor

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

std::size_t parse_size_t(const std::map<std::string, std::string>& params,
                         const std::string& key,
                         std::size_t fallback) {
    auto it = params.find(key);
    if (it == params.end()) {
        return fallback;
    }
    try {
        return static_cast<std::size_t>(std::stoull(it->second));
    } catch (...) {
        return fallback;
    }
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
        group.params.population = parse_size_t(group_cfg.parameters, "population", group.params.population);
        group.params.iteration_limit =
            parse_size_t(group_cfg.parameters, "iterations", group.params.iteration_limit);
        group.params.mutation_rate = parse_double(group_cfg.parameters, "mutation", group.params.mutation_rate);
        group.params.differential_weight =
            parse_double(group_cfg.parameters, "differential", group.params.differential_weight);
        group.params.inertia = parse_double(group_cfg.parameters, "inertia", group.params.inertia);
        group.params.cognitive = parse_double(group_cfg.parameters, "cognitive", group.params.cognitive);
        group.params.social = parse_double(group_cfg.parameters, "social", group.params.social);
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
            runtime->personal_best.store(0.0);

            group.devices.emplace_back(std::move(runtime));
        }

        if (!group.devices.empty()) {
            group.global_best_weights.resize(group.devices.size(), 1.0);
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
    if (upper == "LOADBALANCE" || upper == "LB" || upper == "LOAD-BALANCE") {
        return StrategyType::LoadBalance;
    }
    if (upper == "GA" || upper == "GENETIC") {
        return StrategyType::Genetic;
    }
    if (upper == "DE" || upper == "DIFFERENTIAL") {
        return StrategyType::Differential;
    }
    if (upper == "PSO" || upper == "PARTICLE") {
        return StrategyType::Particle;
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
    case StrategyType::LoadBalance:
        return select_load_balance(group);
    case StrategyType::Genetic:
    case StrategyType::Differential:
    case StrategyType::Particle:
        return select_weighted(group);
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

std::shared_ptr<XschedCore::DeviceRuntime> XschedCore::select_load_balance(DeviceGroup& group) {
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

std::shared_ptr<XschedCore::DeviceRuntime> XschedCore::select_weighted(DeviceGroup& group) {
    if (group.devices.empty()) {
        return nullptr;
    }

    std::vector<double> weights;
    weights.reserve(group.devices.size());
    for (const auto& device : group.devices) {
        double weight = device->weight.load(std::memory_order_relaxed);
        if (!std::isfinite(weight) || weight <= 0.0) {
            weight = 1.0;
        }
        weights.push_back(weight);
    }

    std::discrete_distribution<std::size_t> distribution(weights.begin(), weights.end());
    auto index = distribution(group.rng) % group.devices.size();
    return group.devices[index];
}

void XschedCore::update_strategy(DeviceGroup& group, DeviceRuntime& device, double latency_ms) {
    auto new_avg = device.moving_avg.load(std::memory_order_relaxed);
    if (new_avg <= 0.0) {
        new_avg = latency_ms;
    } else {
        new_avg = (1.0 - kAlpha) * new_avg + kAlpha * latency_ms;
    }
    device.moving_avg.store(new_avg, std::memory_order_relaxed);

    {
        std::lock_guard<std::mutex> history_lock(device.history_mutex);
        device.history.push_back(latency_ms);
        while (device.history.size() > group.params.population) {
            device.history.pop_front();
        }
    }

    double history_avg = 0.0;
    {
        std::lock_guard<std::mutex> history_lock(device.history_mutex);
        if (!device.history.empty()) {
            for (double sample : device.history) {
                history_avg += sample;
            }
            history_avg /= static_cast<double>(device.history.size());
        } else {
            history_avg = new_avg;
        }
    }

    const double fitness = history_avg > kEpsilon ? 1.0 / (history_avg + kEpsilon) : 1.0;
    auto personal_best = device.personal_best.load(std::memory_order_relaxed);
    if (fitness > personal_best) {
        device.personal_best.store(fitness, std::memory_order_relaxed);
        personal_best = fitness;
    }

    switch (group.params.type) {
    case StrategyType::RoundRobin:
    case StrategyType::LoadBalance:
        device.weight.store(std::max(fitness, kEpsilon), std::memory_order_relaxed);
        break;
    case StrategyType::Genetic: {
        const double current_weight = device.weight.load(std::memory_order_relaxed);
        const double mutated = 0.7 * current_weight + 0.3 * fitness * (1.0 + group.params.mutation_rate);
        device.weight.store(std::max(mutated, kEpsilon), std::memory_order_relaxed);
        break;
    }
    case StrategyType::Differential: {
        if (group.devices.size() >= 3) {
            std::uniform_int_distribution<std::size_t> dist(0, group.devices.size() - 1);
            std::size_t a = dist(group.rng);
            std::size_t b = dist(group.rng);
            std::size_t c = dist(group.rng);
            while (a == b) b = dist(group.rng);
            while (c == a || c == b) c = dist(group.rng);
            const double weight_a = group.devices[a]->weight.load(std::memory_order_relaxed);
            const double weight_b = group.devices[b]->weight.load(std::memory_order_relaxed);
            const double weight_c = group.devices[c]->weight.load(std::memory_order_relaxed);
            const double mutated = weight_a + group.params.differential_weight * (weight_b - weight_c);
            device.weight.store(std::max(mutated, kEpsilon), std::memory_order_relaxed);
        } else {
            device.weight.store(std::max(fitness, kEpsilon), std::memory_order_relaxed);
        }
        break;
    }
    case StrategyType::Particle: {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double weight = device.weight.load(std::memory_order_relaxed);
        double velocity = device.velocity.load(std::memory_order_relaxed);
        const double global_best = group.global_best;
        velocity = group.params.inertia * velocity +
                   group.params.cognitive * dist(group.rng) * (personal_best - weight) +
                   group.params.social * dist(group.rng) * (global_best - weight);
        weight = std::max(weight + velocity, kEpsilon);
        device.velocity.store(velocity, std::memory_order_relaxed);
        device.weight.store(weight, std::memory_order_relaxed);
        break;
    }
    default:
        break;
    }

    if (fitness > group.global_best) {
        group.global_best = fitness;
        group.global_best_weights.clear();
        group.global_best_weights.reserve(group.devices.size());
        for (const auto& dev : group.devices) {
            group.global_best_weights.push_back(dev->weight.load(std::memory_order_relaxed));
        }
    }
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

    std::lock_guard<std::mutex> lock(m_pending_mutex);
    m_pending.erase(pending.command);
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

}  // namespace template_plugin
}  // namespace ov
