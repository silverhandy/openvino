// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "openvino/core/node.hpp"
#include "openvino/itt.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/ivariable_state.hpp"
#include "openvino/runtime/profiling_info.hpp"

namespace ov {
namespace template_plugin {

// forward declaration
class CompiledModel;
class XschedCore;
struct XschedPendingCommand;

// ! [infer_request:header]
class InferRequest : public ov::ISyncInferRequest {
public:
    explicit InferRequest(const std::shared_ptr<const ov::template_plugin::CompiledModel>& compiled_model);
    ~InferRequest();

    void infer() override;
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

    // pipeline methods-stages which are used in async infer request implementation and assigned to particular executor
    void infer_preprocess();
    void start_pipeline();
    void wait_pipeline();
    void infer_postprocess();
    void cancel();

    void set_tensors_impl(const ov::Output<const ov::Node> port,
                          const std::vector<ov::SoPtr<ov::ITensor>>& tensors) override;

private:
    friend class XschedCore;

    void execute_on_device(XschedPendingCommand& pending);
    void finalize_pending(const XschedPendingCommand& pending);
    void reset_profiling();

    std::shared_ptr<const CompiledModel> get_template_model() const;

    enum { Preprocess, Postprocess, StartPipeline, WaitPipeline, numOfStages };

    std::array<openvino::itt::handle_t, numOfStages> m_profiling_task;
    std::array<std::chrono::steady_clock::duration, numOfStages> m_stage_durations;

    std::shared_ptr<XschedCore> m_runtime;
    std::shared_ptr<XschedPendingCommand> m_pending;
    std::mutex m_pending_mutex;
    std::atomic<bool> m_cancelled{false};
    std::vector<ov::SoPtr<ov::IVariableState>> m_variable_states;
    std::vector<ov::ProfilingInfo> m_last_profiling;
};
// ! [infer_request:header]

}  // namespace template_plugin
}  // namespace ov
