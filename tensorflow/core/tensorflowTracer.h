#undef TRACEPOINT_PROVIDER
#define TRACEPOINT_PROVIDER tensorflowTracer

#undef TRACEPOINT_INCLUDE
#define TRACEPOINT_INCLUDE "./tensorflowTracer.h"

#if !defined(_TENSORFLOWTRACER_H) || defined(TRACEPOINT_HEADER_MULTI_READ)
#define _TENSORFLOWTRACER_H

#include <lttng/tracepoint.h>


// Tracepoints : entry / exit

// TensorFlow scheduler
TRACEPOINT_EVENT(
    tensorflowTracer,
    process_entry,
    TP_ARGS(
        const char*, cat_arg,
        const char*, name_arg,
        uint64_t, schedule_arg
    ),
    TP_FIELDS(
        ctf_string(cat, cat_arg)
        ctf_string(name, name_arg)
        ctf_integer(uint64_t, schedule, schedule_arg)
    )
)
TRACEPOINT_EVENT(
    tensorflowTracer,
    process_exit,
    TP_ARGS(
        const char*, cat_arg,
        const char*, name_arg,
        uint64_t, schedule_arg
    ),
    TP_FIELDS(
        ctf_string(cat, cat_arg)
        ctf_string(name, name_arg)
        ctf_integer(uint64_t, schedule, schedule_arg)
    )
)

TRACEPOINT_EVENT(
    tensorflowTracer,
    inline_ready_entry,
    TP_ARGS(
        const char*, cat_arg,
        const char*, name_arg
    ),
    TP_FIELDS(
        ctf_string(cat, cat_arg)
        ctf_string(name, name_arg)
    )
)
TRACEPOINT_EVENT(
    tensorflowTracer,
    inline_ready_exit,
    TP_ARGS(
        const char*, cat_arg,
        const char*, name_arg
    ),
    TP_FIELDS(
        ctf_string(cat, cat_arg)
        ctf_string(name, name_arg)
    )
)

TRACEPOINT_EVENT(
    tensorflowTracer,
    push_succ_entry,
    TP_ARGS(
        const char*, cat_arg,
        const char*, name_arg
    ),
    TP_FIELDS(
        ctf_string(cat, cat_arg)
        ctf_string(name, name_arg)
    )
)
TRACEPOINT_EVENT(
    tensorflowTracer,
    push_succ_exit,
    TP_ARGS(
        const char*, cat_arg,
        const char*, name_arg,
        int, is_ready_arg
    ),
    TP_FIELDS(
        ctf_string(cat, cat_arg)
        ctf_string(name, name_arg)
        ctf_integer(int, is_ready, is_ready_arg)
    )
)


// Tracepoints : start / end
// TensorFlow session
TRACEPOINT_EVENT(
    tensorflowTracer,
    session_start,
    TP_ARGS(
        const char*, cat_arg,
        const char*, name_arg,
        int, count_arg
    ),
    TP_FIELDS(
        ctf_string(cat, cat_arg)
        ctf_string(name, name_arg)
        ctf_integer(int, count, count_arg)
    )
)
TRACEPOINT_EVENT(
    tensorflowTracer,
    session_end,
    TP_ARGS(
        const char*, cat_arg,
        const char*, name_arg,
        int, count_arg
    ),
    TP_FIELDS(
        ctf_string(cat, cat_arg)
        ctf_string(name, name_arg)
        ctf_integer(int, count, count_arg)
    )
)

// TensorFlow sync operations
TRACEPOINT_EVENT(
    tensorflowTracer,
    operation_start,
    TP_ARGS(
        const char*, cat_arg,
        const char*, placement_arg,
        const char*, name_arg
    ),
    TP_FIELDS(
        ctf_string(cat, cat_arg)
        ctf_string(placement, placement_arg)
        ctf_string(name, name_arg)
    )
)
TRACEPOINT_EVENT(
    tensorflowTracer,
    operation_end,
    TP_ARGS(
        const char*, cat_arg,
        const char*, placement_arg,
        const char*, name_arg
    ),
    TP_FIELDS(
        ctf_string(cat, cat_arg)
        ctf_string(placement, placement_arg)
        ctf_string(name, name_arg)
    )
)

// TensorFlow async operations
TRACEPOINT_EVENT(
    tensorflowTracer,
    async_operation_start,
    TP_ARGS(
        const char*, cat_arg,
        const char*, placement_arg,
        const char*, name_arg
    ),
    TP_FIELDS(
        ctf_string(cat, cat_arg)
        ctf_string(placement, placement_arg)
        ctf_string(name, name_arg)
    )
)
TRACEPOINT_EVENT(
    tensorflowTracer,
    async_operation_end,
    TP_ARGS(
        const char*, cat_arg,
        const char*, placement_arg,
        const char*, name_arg
    ),
    TP_FIELDS(
        ctf_string(cat, cat_arg)
        ctf_string(placement, placement_arg)
        ctf_string(name, name_arg)
    )
)

// TensorFlow send / recv operations with Rendezvous
TRACEPOINT_EVENT(
    tensorflowTracer,
    rdv_send,
    TP_ARGS(
        const char*, cat_arg,
        const char*, name_arg
    ),
    TP_FIELDS(
        ctf_string(cat, cat_arg)
        ctf_string(name, name_arg)
    )
)
TRACEPOINT_EVENT(
    tensorflowTracer,
    rdv_recv,
    TP_ARGS(
        const char*, cat_arg,
        const char*, name_arg
    ),
    TP_FIELDS(
        ctf_string(cat, cat_arg)
        ctf_string(name, name_arg)
    )
)


// Tracepoints : XY chart
// BFC allocator statistics
TRACEPOINT_EVENT(
    tensorflowTracer,
    bfc_allocator_stats,
    TP_ARGS(
        const char*, allocator_name_arg,
        uint64_t, num_allocs_arg,
        uint64_t, bytes_in_use_arg,
        uint64_t, max_bytes_in_use_arg,
        uint64_t, max_alloc_size_arg
    ),
    TP_FIELDS(
        ctf_string(allocator_name, allocator_name_arg)
        ctf_integer(uint64_t, num_allocs, num_allocs_arg)
        ctf_integer(uint64_t, bytes_in_use, bytes_in_use_arg)
        ctf_integer(uint64_t, max_bytes_in_use, max_bytes_in_use_arg)
        ctf_integer(uint64_t, max_alloc_size, max_alloc_size_arg)
    )
)

// BFC allocator, chunks statistics
TRACEPOINT_EVENT(
    tensorflowTracer,
    bfc_chunks_stats,
    TP_ARGS(
        const char*, allocator_name_arg,
        uint64_t, total_bytes_in_use_arg,
        uint64_t, total_requested_bytes_in_use_arg,
        uint64_t, total_wasted_bytes_in_use_arg,
        uint64_t, total_bytes_arg,
        uint64_t, total_requested_bytes_arg,
        uint64_t, total_wasted_bytes_arg,
        uint64_t, chunks_arg,
        uint64_t, in_use_chunks_arg,
        uint64_t, free_chunks_arg
    ),
    TP_FIELDS(
        ctf_string(allocator_name, allocator_name_arg)
        ctf_integer(int64_t, total_bytes_in_use, total_bytes_in_use_arg)
        ctf_integer(int64_t, total_requested_bytes_in_use, total_requested_bytes_in_use_arg)
        ctf_integer(int64_t, total_wasted_bytes_in_use, total_wasted_bytes_in_use_arg)
        ctf_integer(int64_t, total_bytes, total_bytes_arg)
        ctf_integer(int64_t, total_requested_bytes, total_requested_bytes_arg)
        ctf_integer(int64_t, total_wasted_bytes, total_wasted_bytes_arg)
        ctf_integer(int64_t, chunks, chunks_arg)
        ctf_integer(int64_t, in_use_chunks, in_use_chunks_arg)
        ctf_integer(int64_t, free_chunks, free_chunks_arg)
    )
)

// BFC allocator, bins statistics
TRACEPOINT_EVENT(
    tensorflowTracer,
    bfc_bins_stats,
    TP_ARGS(
        const char*, allocator_name_arg,
        uint64_t, bin_numero_arg,
        uint64_t, total_chunks_in_bin_arg,
        uint64_t, total_chunks_in_use_arg,
        uint64_t, total_bytes_in_bin_arg,
        uint64_t, total_bytes_in_use_arg,
        uint64_t, total_requested_bytes_in_use_arg
    ),
    TP_FIELDS(
        ctf_string(allocator_name, allocator_name_arg)
        ctf_integer(int64_t, bin_numero, bin_numero_arg)
        ctf_integer(int64_t, total_chunks_in_bin, total_chunks_in_bin_arg)
        ctf_integer(int64_t, total_chunks_in_use, total_chunks_in_use_arg)
        ctf_integer(int64_t, total_bytes_in_bin, total_bytes_in_bin_arg)
        ctf_integer(int64_t, total_bytes_in_use, total_bytes_in_use_arg)
        ctf_integer(int64_t, total_requested_bytes_in_use, total_requested_bytes_in_use_arg)
    )
)

// GPU driver memory allocation / deallocation
TRACEPOINT_EVENT(
    tensorflowTracer,
    memory_allocate,
    TP_ARGS(
        const char*, device_arg,
        const char*, ptr_arg,
        uint64_t, bytes_arg
    ),
    TP_FIELDS(
        ctf_string(device, device_arg)
        ctf_string(ptr, ptr_arg)
        ctf_integer(int64_t, bytes, bytes_arg)
    )
)
TRACEPOINT_EVENT(
    tensorflowTracer,
    memory_deallocate,
    TP_ARGS(
        const char*, device_arg,
        const char*, ptr_arg,
        uint64_t, bytes_arg
    ),
    TP_FIELDS(
        ctf_string(device, device_arg)
        ctf_string(ptr, ptr_arg)
        ctf_integer(int64_t, bytes, bytes_arg)
    )
)



#endif

#include <lttng/tracepoint-event.h>
