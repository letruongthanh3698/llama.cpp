#pragma once

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define RPC_PROTO_MAJOR_VERSION    4
#define RPC_PROTO_MINOR_VERSION    0
#define RPC_PROTO_PATCH_VERSION    1

#ifdef  __cplusplus
static_assert(GGML_OP_COUNT == 97, "GGML_OP_COUNT has changed - update RPC_PROTO_PATCH_VERSION");
#endif

#define GGML_RPC_MAX_SERVERS       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device);
GGML_BACKEND_API bool ggml_backend_is_rpc(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device);

GGML_BACKEND_API void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total);

GGML_BACKEND_API void ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir,
                                                    size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_reg(void);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint);

// -------------------------------------------------------------------------------------------------
// P2P ring transport — decoupled custom-command layer (Phase 7).
//
// The decentralized ring forwards a device's boundary hidden state to its successor. Rather than
// merge the app's semantics into libggml-rpc (which would couple the lib to app headers and, on
// macOS, break weak-symbol hooks across the dylib boundary), the lib exposes a tiny app-agnostic
// layer: the app REGISTERS a handler (a plain function pointer — resolves fine across the lib
// boundary) and SENDS commands keyed by an endpoint string. All wire types are opaque:
// an int32 client_id plus raw payload bytes; the app owns all interpretation.
//
// Command ids live in [GGML_RPC_P2P_CMD_BASE, RPC_CMD_COUNT) and are static_assert'd against the
// internal rpc_cmd enum in ggml-rpc.cpp. Both peers must run the same build.
enum ggml_rpc_p2p_cmd {
    GGML_RPC_P2P_CMD_SET_HIDDEN_STATE = 17,  // predecessor -> device: pre-norm hidden state (fire-and-forget)
    GGML_RPC_P2P_CMD_RETURN_TOKEN     = 18,  // tail -> head: sampled token id (fire-and-forget)
    GGML_RPC_P2P_CMD_SET_PREFILL_DATA = 19,  // prefill node -> decoder node: serialized KV slice (handoff)
};

// Handler invoked on the SERVER for any P2P command. `in`/`in_len` is the payload (client_id already
// stripped). To reply, malloc a buffer into *out and set *out_len (the lib frees it after sending);
// leave *out == NULL for an empty reply.
typedef void (*ggml_rpc_p2p_handler_t)(int32_t cmd, int32_t client_id,
                                       const uint8_t * in, size_t in_len,
                                       uint8_t ** out, size_t * out_len, void * user);

// Register the P2P handler (called from within rpc_serve_client). Pass NULL to unregister.
GGML_BACKEND_API void ggml_rpc_register_p2p_handler(ggml_rpc_p2p_handler_t fn, void * user);

// CLIENT: connect to `endpoint` (host:port) and send a P2P command. On success *out (malloc'd, caller
// frees) holds the response bytes and *out_len its length. Returns false on connect/IO failure.
GGML_BACKEND_API bool ggml_rpc_p2p_send(const char * endpoint, int32_t cmd, int32_t client_id,
                                        const uint8_t * in, size_t in_len,
                                        uint8_t ** out, size_t * out_len);

#ifdef  __cplusplus
}
#endif
