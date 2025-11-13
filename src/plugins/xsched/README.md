# OpenVINO XSched Plugin

The XSched plugin extends OpenVINO™ Runtime with a multi-device scheduler that decides, per inference, which accelerator (CPU, iGPU, NPU, etc.) should execute the request. It builds one compiled model per target device, registers lightweight queues with the xsched runtime, and adapts its dispatch strategy based on live latency feedback.

## Architecture Overview

- **Plugin (`plugin.cpp`)** – exposes the `XSCHED` device to `ov::Core`, parses configuration, and produces compiled models.
- **Compiled model (`compiled_model.cpp`)** – optionally transforms the loaded network, creates the scheduler runtime, and wires profiling into the model.
- **Scheduler runtime (`xsched_core.cpp`)** – groups devices, manages xsched queues, tracks moving-average latencies, and chooses devices via round-robin, least-loaded, or adaptive weighted strategies.
- **Infer requests (`sync_infer_request.cpp`, `async_infer_request.cpp`)** – handle tensor preparation, submit work to xsched, and surface profiling data back to the application.
- **Backend helpers (`backend/`)** – provide the glue between xsched software queues and OpenVINO executable networks.

## Prerequisites

1. A local OpenVINO developer build (2025+).
2. xsched runtime libraries (bundled under `3rd-party/xsched`).
3. **A running xsched server** (`xsched-server`). The plugin communicates with the scheduler over the default channel `xsched-server`; compilation or inference will fail if the server is absent.

### Starting the xsched server

Launch the scheduler before running an XSCHED workload:

```bash
# From repository root – brings up xsched services with the shipped script
./start-server.sh

# Or start the binary produced by the xsched build directory
<xsched_build>/bin/xsched-server --config <path-to-xsched-config>
```

Verify the process is alive (for example, `pgrep xsched-server`) and adjust permissions so the plugin can reach the server’s IPC channel.

## Building the Plugin

```bash
# Build or reuse an OpenVINO developer package
cmake -S <openvino_src> -B <openvino_build> -DENABLE_TESTS=ON -DENABLE_FUNCTIONAL_TESTS=ON
cmake --build <openvino_build> --target all -j

# Configure and build the xsched plugin against that package
cmake -S 3rd-party/openvino/src/plugins/xsched \
	  -B 3rd-party/openvino/src/plugins/xsched/build \
	  -DOpenVINODeveloperPackage_DIR=<openvino_build> \
	  -DENABLE_TEMPLATE_REGISTRATION=ON
cmake --build 3rd-party/openvino/src/plugins/xsched/build --target install -j
```

The install step registers the plugin entry in `plugins.xml`. For manual deployment copy the generated `*.so`/`*.dll` and `plugin.xml` to a directory referenced by `OV_PLUGIN_PATH` or drop them into the OpenVINO installation’s `runtime/lib/intel64` folder.

## Deployment & Usage

1. Start `xsched-server` (see **Prerequisites**).
2. Optionally export environment variables for the runtime, e.g. `export XSCHED_SERVER_CHANNEL_NAME=xsched-server` if you override the default channel.
3. Execute your OpenVINO workload targeting the `XSCHED` device:

```python
import openvino as ov

core = ov.Core()
model = core.read_model("/path/to/model.xml")

compiled = core.compile_model(model, "XSCHED", {
	"DEVICE_PRIORITIES": "NPU,GPU.0,CPU",
	"ENABLE_PROFILING": True,
})

infer_request = compiled.create_infer_request()
infer_request.infer()
print(infer_request.get_profiling_info())
```

4. Advanced knobs are exposed through configuration maps:
   - `DEVICE_PRIORITY_GROUPS` – declare multiple device groups, each with its own strategy (e.g. `adaptive`) and smoothing bounds.
   - `NUM_STREAMS` / `PERFORMANCE_HINT` – control the plugin’s internal parallelism.
   - `ENABLE_PROFILING` – capture end-to-end queue wait, device execution, and total latency metrics.

## Troubleshooting

- **Connection errors** – make sure `xsched-server` is running and reachable; the plugin retries neither discovery nor reconnection.
- **No device selected** – confirm the requested device IDs appear in `core.get_available_devices()` and that the xsched server advertises matching devices.
- **Suboptimal scheduling** – switch the strategy to `adaptive` and tune `smoothing`, `min_weight`, and `max_weight` parameters to weigh latency history appropriately.

## Further Reading

- [OpenVINO Plugin Developer Guide](https://docs.openvino.ai/2025/documentation/openvino-extensibility/openvino-plugin-library.html)
- [xsched project](https://github.com/intel/xsched) – runtime design, server configuration, and diagnostics.
- Project-wide references: [OpenVINO™ README](../../../README.md), [Plugin index](../README.md).
