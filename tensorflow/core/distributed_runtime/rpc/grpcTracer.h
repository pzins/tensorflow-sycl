#undef TRACEPOINT_PROVIDER
#define TRACEPOINT_PROVIDER grpcTracer

#undef TRACEPOINT_INCLUDE
#define TRACEPOINT_INCLUDE "./grpcTracer.h"

#if !defined(_grpcTracer_H) || defined(TRACEPOINT_HEADER_MULTI_READ)
#define _grpcTracer_H

#include <lttng/tracepoint.h>
TRACEPOINT_EVENT(
    grpcTracer,
    EncodeRecvTensorResponseToByteBuffer,
    TP_ARGS(
        const char*, cat_arg,
        const char*, name_arg,
        uint64_t, size_arg
    ),
    TP_FIELDS(
        ctf_string(cat, cat_arg)
        ctf_string(name, name_arg)
        ctf_integer(uint64_t, size, size_arg)
    )
)

TRACEPOINT_EVENT(
    grpcTracer,
    EncodeTensorToByteBuffer,
    TP_ARGS(
        const char*, cat_arg,
        const char*, name_arg,
        uint64_t, size_arg
    ),
    TP_FIELDS(
        ctf_string(cat, cat_arg)
        ctf_string(name, name_arg)
        ctf_integer(uint64_t, size, size_arg)
    )
)

// Sending and receiving request tracepoint
// For GetStatusAsync, RegisterGraphAsync, DeregisterGraphAsync, RunGraphAsync, 
// CleanupGraphAsync, CleanupAllAsync, LoggingAsync and TracingAsync but not 
// RecvTensorAsync
TRACEPOINT_EVENT(
    grpcTracer,
    send_request,
    TP_ARGS(
        const char*, name_arg
    ),
    TP_FIELDS(
        ctf_string(name, name_arg)
    )
)
TRACEPOINT_EVENT(
    grpcTracer,
    receive_request,
    TP_ARGS(
        const char*, name_arg
    ),
    TP_FIELDS(
        ctf_string(name, name_arg)
    )
)

// Sending and receiving request tracepoint for RecvTensorAsync
TRACEPOINT_EVENT(
    grpcTracer,
    send_RecvTensor_request,
    TP_ARGS(
        const char*, cat_arg,
        const char*, name_arg,
        const char*, rendezvous_key_arg,
        const char*, tensor_arg,
        const char*, src_device_arg,
        const char*, dst_device_arg,
        const char*, request_arg,
        const char*, response_arg
    ),
    TP_FIELDS(
        ctf_string(cat, cat_arg)
        ctf_string(name, name_arg)
        ctf_string(rendezvous_key, rendezvous_key_arg)
        ctf_string(tensor, tensor_arg)
        ctf_string(src_device, src_device_arg)
        ctf_string(dst_device, dst_device_arg)
        ctf_string(request, request_arg)
        ctf_string(response, response_arg)
    )
)
TRACEPOINT_EVENT(
    grpcTracer,
    receive_RecvTensor_request,
    TP_ARGS(
        const char*, cat_arg,
        const char*, name_arg,
        const char*, rendezvous_key_arg,
        uint64_t, step_id_arg,
        uint32_t, bus_id_arg
    ),
    TP_FIELDS(
        ctf_string(cat, cat_arg)
        ctf_string(name, name_arg)
        ctf_string(rendezvous_key, rendezvous_key_arg)
        ctf_integer(uint64_t, step_id, step_id_arg)
        ctf_integer(uint32_t, bus_id, bus_id_arg)
    )
)

// Describing the time taken to get a protobuf (CPU RAM) from a GPU Tensor
TRACEPOINT_EVENT(
    grpcTracer,
    set_proto_from_gpu_start,
    TP_ARGS(
        const char*, cat_arg,
        const char*, name_arg,
        const char*, rendezvous_key_arg
    ),
    TP_FIELDS(
        ctf_string(cat, cat_arg)
        ctf_string(name, name_arg)
        ctf_string(rendezvous_key, rendezvous_key_arg)
    )
)
TRACEPOINT_EVENT(
    grpcTracer,
    set_proto_from_gpu_end,
    TP_ARGS(
        const char*, cat_arg,
        const char*, name_arg,
        const char*, rendezvous_key_arg
    ),
    TP_FIELDS(
        ctf_string(cat, cat_arg)
        ctf_string(name, name_arg)
        ctf_string(rendezvous_key, rendezvous_key_arg)
    )
)

// Describing the duration between the recption of a tensor request and the 
// response with the tensor value 
TRACEPOINT_EVENT(
    grpcTracer,
    prepare_response_tensor_start,
    TP_ARGS(
        const char*, cat_arg,
        const char*, name_arg,
        const char*, rendezvous_key_arg
    ),
    TP_FIELDS(
        ctf_string(cat, cat_arg)
        ctf_string(name, name_arg)
        ctf_string(rendezvous_key, rendezvous_key_arg)
    )
)
TRACEPOINT_EVENT(
    grpcTracer,
    prepare_response_tensor_end,
    TP_ARGS(
        const char*, cat_arg,
        const char*, name_arg,
        const char*, rendezvous_key_arg
    ),
    TP_FIELDS(
        ctf_string(cat, cat_arg)
        ctf_string(name, name_arg)
        ctf_string(rendezvous_key, rendezvous_key_arg)
    )
)

// Describing the duration between the sending request for a tensor
// and the moment when the response arrives
TRACEPOINT_EVENT(
    grpcTracer,
    send_request_tensor_start,
    TP_ARGS(
        const char*, cat_arg,
        const char*, name_arg,
        const char*, rendezvous_key_arg
    ),
    TP_FIELDS(
        ctf_string(cat, cat_arg)
        ctf_string(name, name_arg)
        ctf_string(rendezvous_key, rendezvous_key_arg)
    )
)
TRACEPOINT_EVENT(
    grpcTracer,
    send_request_tensor_end,
    TP_ARGS(
        const char*, cat_arg,
        const char*, name_arg,
        const char*, rendezvous_key_arg
    ),
    TP_FIELDS(
        ctf_string(cat, cat_arg)
        ctf_string(name, name_arg)
        ctf_string(rendezvous_key, rendezvous_key_arg)
    )
)

#endif

#include <lttng/tracepoint-event.h>
