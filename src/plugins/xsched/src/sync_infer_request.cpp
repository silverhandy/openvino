// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sync_infer_request.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "itt.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "openvino/runtime/profiling_info.hpp"
#include "compiled_model.hpp"
#include "xsched_core.hpp"

using Time = std::chrono::steady_clock;

namespace {

ov::ProfilingInfo make_profiling_info(const std::string& name, std::chrono::steady_clock::duration duration) {
    ov::ProfilingInfo info;
    info.status = ov::ProfilingInfo::Status::EXECUTED;
    info.node_name = name;
    info.cpu_time = info.real_time = std::chrono::duration_cast<std::chrono::microseconds>(duration);
    return info;
}

}  // namespace

// ! [infer_request:ctor]
ov::template_plugin::InferRequest::InferRequest(const std::shared_ptr<const ov::template_plugin::CompiledModel>& model)
    : ov::ISyncInferRequest(std::static_pointer_cast<const ov::ICompiledModel>(model)) {
    auto compiled = get_template_model();
    auto request_id = std::to_string(compiled->m_request_id.fetch_add(1));
    std::string name = compiled->m_model->get_friendly_name() + "_Req" + request_id;

    m_profiling_task = {openvino::itt::handle("XSCHED_" + name + "_Preprocess"),
                        openvino::itt::handle("XSCHED_" + name + "_Postprocess"),
                        openvino::itt::handle("XSCHED_" + name + "_Schedule"),
                        openvino::itt::handle("XSCHED_" + name + "_Wait")};
    m_stage_durations.fill(std::chrono::steady_clock::duration::zero());

    m_runtime = compiled->get_runtime();
    OPENVINO_ASSERT(m_runtime, "XSCHED runtime is not initialized");

    // Pre-allocate tensors to match model signature
    for (const auto& input : get_inputs()) {
        allocate_tensor(input, [input](ov::SoPtr<ov::ITensor>& tensor) {
            auto element = input.get_element_type();
            auto shape = input.get_partial_shape().is_dynamic() ? ov::Shape{0} : input.get_shape();
            if (!tensor || tensor->get_element_type() != element) {
                tensor = ov::make_tensor(element, shape);
            } else {
                tensor->set_shape(shape);
            }
        });
    }
    for (const auto& output : get_outputs()) {
        allocate_tensor(output, [output](ov::SoPtr<ov::ITensor>& tensor) {
            auto element = output.get_element_type();
            auto shape = output.get_partial_shape().is_dynamic() ? ov::Shape{0} : output.get_shape();
            if (!tensor || tensor->get_element_type() != element) {
                tensor = ov::make_tensor(element, shape);
            } else {
                tensor->set_shape(shape);
            }
        });
    }
}
// ! [infer_request:ctor]

// ! [infer_request:dtor]
ov::template_plugin::InferRequest::~InferRequest() = default;
// ! [infer_request:dtor]

// ! [infer_request:set_tensors_impl]
void ov::template_plugin::InferRequest::set_tensors_impl(const ov::Output<const ov::Node> port,
                                                         const std::vector<ov::SoPtr<ov::ITensor>>& tensors) {
    for (const auto& input : get_inputs()) {
        if (input == port) {
            m_batched_tensors[input.get_tensor_ptr()] = tensors;
            return;
        }
    }
    OPENVINO_THROW("Cannot find input tensors for port ", port);
}
// ! [infer_request:set_tensors_impl]

// ! [infer_request:query_state]
std::vector<ov::SoPtr<ov::IVariableState>> ov::template_plugin::InferRequest::query_state() const {
    return m_variable_states;
}
// ! [infer_request:query_state]

std::shared_ptr<const ov::template_plugin::CompiledModel> ov::template_plugin::InferRequest::get_template_model()
    const {
    auto& compiled_model = get_compiled_model();
    auto template_model = std::dynamic_pointer_cast<const ov::template_plugin::CompiledModel>(compiled_model);
    OPENVINO_ASSERT(template_model);
    return template_model;
}

// ! [infer_request:infer]
void ov::template_plugin::InferRequest::infer() {
    infer_preprocess();
    start_pipeline();
    wait_pipeline();
    infer_postprocess();
}
// ! [infer_request:infer]

void ov::template_plugin::InferRequest::reset_profiling() {
    m_stage_durations.fill(std::chrono::steady_clock::duration::zero());
    m_last_profiling.clear();
}

// ! [infer_request:infer_preprocess]
void ov::template_plugin::InferRequest::infer_preprocess() {
    OV_ITT_SCOPED_TASK(itt::domains::XSchedPlugin, m_profiling_task[Preprocess]);
    auto start = Time::now();
    reset_profiling();
    m_cancelled.store(false, std::memory_order_relaxed);
    convert_batched_tensors();
    check_tensors();
    m_stage_durations[Preprocess] = Time::now() - start;
}
// ! [infer_request:infer_preprocess]

// ! [infer_request:start_pipeline]
void ov::template_plugin::InferRequest::start_pipeline() {
    OV_ITT_SCOPED_TASK(itt::domains::XSchedPlugin, m_profiling_task[StartPipeline]);
    auto start = Time::now();
    auto new_pending = m_runtime->enqueue(*this);
    std::shared_ptr<XschedPendingCommand> pending;
    {
        std::lock_guard<std::mutex> lock(m_pending_mutex);
        m_pending = new_pending;
        pending = m_pending;
    }
    OPENVINO_ASSERT(pending, "XSCHED runtime failed to create pending command");
    m_stage_durations[StartPipeline] = Time::now() - start;
}
// ! [infer_request:start_pipeline]

// ! [infer_request:wait_pipeline]
void ov::template_plugin::InferRequest::wait_pipeline() {
    OV_ITT_SCOPED_TASK(itt::domains::XSchedPlugin, m_profiling_task[WaitPipeline]);
    auto start = Time::now();
    std::shared_ptr<XschedPendingCommand> pending;
    {
        std::lock_guard<std::mutex> lock(m_pending_mutex);
        pending = m_pending;
        if (pending) {
            m_pending.reset();
        }
    }
    if (pending) {
        try {
            pending->future.get();
        } catch (...) {
            m_stage_durations[WaitPipeline] = Time::now() - start;
            {
                std::lock_guard<std::mutex> lock(m_pending_mutex);
                m_pending.reset();
            }
            throw;
        }
    }
    m_stage_durations[WaitPipeline] = Time::now() - start;
}
// ! [infer_request:wait_pipeline]

// ! [infer_request:infer_postprocess]
void ov::template_plugin::InferRequest::infer_postprocess() {
    OV_ITT_SCOPED_TASK(itt::domains::XSchedPlugin, m_profiling_task[Postprocess]);
    auto start = Time::now();

    std::shared_ptr<XschedPendingCommand> pending;
    {
        std::lock_guard<std::mutex> lock(m_pending_mutex);
        pending = m_pending;
        m_pending.reset();
    }

    if (pending) {
        finalize_pending(*pending);
    }

    m_stage_durations[Postprocess] = Time::now() - start;
}
// ! [infer_request:infer_postprocess]

// ! [infer_request:get_profiling_info]
std::vector<ov::ProfilingInfo> ov::template_plugin::InferRequest::get_profiling_info() const {
    return m_last_profiling;
}
// ! [infer_request:get_profiling_info]

// ! [infer_request:cancel]
void ov::template_plugin::InferRequest::cancel() {
    m_cancelled.store(true, std::memory_order_relaxed);
    std::shared_ptr<XschedPendingCommand> pending;
    {
        std::lock_guard<std::mutex> lock(m_pending_mutex);
        pending = m_pending;
    }
    if (pending) {
        try {
            pending->error = std::make_exception_ptr(std::runtime_error("Inference was cancelled"));
            pending->completion.set_exception(pending->error);
        } catch (...) {
        }
    }
}
// ! [infer_request:cancel]

void ov::template_plugin::InferRequest::execute_on_device(XschedPendingCommand& pending) {
    if (m_cancelled.load(std::memory_order_relaxed)) {
        OPENVINO_THROW("Inference was cancelled");
    }

    OPENVINO_ASSERT(pending.device, "XSCHED pending command missing target device");
    auto device_request = pending.device->compiled_model->create_infer_request();
    OPENVINO_ASSERT(device_request, "XSCHED device compiled model returned null infer request");

    const auto& inputs = get_inputs();
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto tensor = get_tensor(inputs[i]);
        device_request->set_tensor(inputs[i], tensor);
    }

    std::vector<std::pair<ov::Output<const ov::Node>, ov::SoPtr<ov::ITensor>>> copy_back;
    const auto& outputs = get_outputs();
    for (size_t i = 0; i < outputs.size(); ++i) {
        auto tensor = get_tensor(outputs[i]);
        try {
            device_request->set_tensor(outputs[i], tensor);
        } catch (const ov::Exception&) {
            copy_back.emplace_back(outputs[i], tensor);
        }
    }

    device_request->infer();

    for (auto& entry : copy_back) {
        auto device_tensor = device_request->get_tensor(entry.first);
        if (device_tensor && entry.second) {
            device_tensor->copy_to(entry.second._ptr);
        }
    }

    if (pending.enable_profiling) {
        pending.profiling.clear();
        try {
            auto internal = device_request->get_profiling_info();
            pending.profiling.insert(pending.profiling.end(), internal.begin(), internal.end());
        } catch (...) {
        }
    }
}

void ov::template_plugin::InferRequest::finalize_pending(const XschedPendingCommand& pending) {
    std::vector<ov::ProfilingInfo> info;
    info.reserve(m_stage_durations.size() + pending.profiling.size());
    info.emplace_back(make_profiling_info("xsched.preprocess", m_stage_durations[Preprocess]));
    info.emplace_back(make_profiling_info("xsched.schedule", m_stage_durations[StartPipeline]));
    info.emplace_back(make_profiling_info("xsched.wait", m_stage_durations[WaitPipeline]));
    info.emplace_back(make_profiling_info("xsched.postprocess", m_stage_durations[Postprocess]));
    info.insert(info.end(), pending.profiling.begin(), pending.profiling.end());
    m_last_profiling = std::move(info);
    m_cancelled.store(false, std::memory_order_relaxed);
}

