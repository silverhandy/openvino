// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "async_infer_request.hpp"

#include "itt.hpp"
#include "openvino/runtime/iinfer_request.hpp"
#include "sync_infer_request.hpp"

// ! [async_infer_request:ctor]
ov::xsched_plugin::AsyncInferRequest::AsyncInferRequest(
    const std::shared_ptr<ov::xsched_plugin::InferRequest>& request,
    const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
    const std::shared_ptr<ov::threading::ITaskExecutor>& wait_executor,
    const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor)
    : ov::IAsyncInferRequest(request, task_executor, callback_executor),
      m_wait_executor(wait_executor) {
    m_cancel_callback = [request] {
        request->cancel();
    };

    m_pipeline = {{task_executor,
                   [request] {
                       OV_ITT_SCOPED_TASK(itt::domains::XSchedPlugin,
                                          "XSCHED::AsyncInferRequest::preprocess_and_schedule");
                       request->infer_preprocess();
                       request->start_pipeline();
                   }},
                  {m_wait_executor,
                   [request] {
                       OV_ITT_SCOPED_TASK(itt::domains::XSchedPlugin,
                                          "XSCHED::AsyncInferRequest::wait_pipeline");
                       request->wait_pipeline();
                   }},
                  {task_executor,
                   [request] {
                       OV_ITT_SCOPED_TASK(itt::domains::XSchedPlugin,
                                          "XSCHED::AsyncInferRequest::postprocess");
                       request->infer_postprocess();
                   }}};
}
// ! [async_infer_request:ctor]

// ! [async_infer_request:dtor]
ov::xsched_plugin::AsyncInferRequest::~AsyncInferRequest() {
    ov::IAsyncInferRequest::stop_and_wait();
}
// ! [async_infer_request:dtor]

// ! [async_infer_request:cancel]
void ov::xsched_plugin::AsyncInferRequest::cancel() {
    ov::IAsyncInferRequest::cancel();
    m_cancel_callback();
}
// ! [async_infer_request:cancel]
