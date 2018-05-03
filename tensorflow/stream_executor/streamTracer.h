#undef TRACEPOINT_PROVIDER
#define TRACEPOINT_PROVIDER streamTracer

#undef TRACEPOINT_INCLUDE
#define TRACEPOINT_INCLUDE "./streamTracer.h"

#if !defined(_streamTracer_H) || defined(TRACEPOINT_HEADER_MULTI_READ)
#define _streamTracer_H

#include <lttng/tracepoint.h>

TRACEPOINT_EVENT(
    streamTracer,
    memcpy_start,
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
    streamTracer,
    memcpy_end,
    TP_ARGS(
        const char*, cat_arg,
        const char*, name_arg
    ),
    TP_FIELDS(
        ctf_string(cat, cat_arg)
        ctf_string(name, name_arg)
    )
)
#endif

#include <lttng/tracepoint-event.h>
