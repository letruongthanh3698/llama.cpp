#include "models.h"

#include <algorithm>   // std::min (BENCH: n_run cap)

void llama_model_qwen2::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    switch (hparams.n_layer()) {
        case 24: type = hparams.n_embd == 1024 ? LLM_TYPE_0_5B : LLM_TYPE_1B; break;
        case 28: type = hparams.n_embd == 1536 ? LLM_TYPE_1_5B : LLM_TYPE_7B; break;
        case 32: type = LLM_TYPE_7B; break;
        case 36: type = LLM_TYPE_3B; break;
        case 40: type = hparams.n_head() == 20 ? LLM_TYPE_4B : LLM_TYPE_13B; break;
        case 48: type = LLM_TYPE_14B; break;
        case 64: type = LLM_TYPE_32B; break;
        case 80: type = LLM_TYPE_70B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen2::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    output_b    = create_tensor(tn(LLM_TENSOR_OUTPUT,      "bias"),   {n_vocab}, TENSOR_NOT_REQUIRED);
    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        // P2P: layers[] is stored compactly for this device's slice, but the GGUF names carry the
        // GLOBAL layer index, so tensor lookups must be by g, not i.
        const int g = (int) hparams.p2p_global_il((uint32_t) i);

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", g), {n_embd}, 0);

        create_tensor_qkv(layer, g, n_embd, n_embd, n_embd_gqa, n_embd_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", g), {n_embd, n_embd}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", g), {n_embd}, 0);

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", g), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", g), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", g), {n_embd,   n_ff}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen2::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_qwen2::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // BENCH: how many blocks to actually run (default = all loaded). 0 blocks isolates embed (head)
    // or finals (tail) on a fully-loaded slice. Inputs used ONLY inside the block loop (positions,
    // attention/KV) must NOT be created when n_run == 0 - an unused input stays unallocated and
    // would fail set_input's buffer assert.
    const int n_run = hparams.p2p_n_active_layers >= 0
        ? std::min<int>(hparams.p2p_n_active_layers, n_layer) : n_layer;

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = n_run > 0 ? build_inp_pos() : nullptr;

    llm_graph_input_attn_kv * inp_attn = n_run > 0 ? build_attn_inp_kv() : nullptr;

    // TAIL only: out_ids gathers the output rows before norm+lm_head. A mid/head ring device
    // exports the full-width hidden state and never gathers, so it must NOT create this input.
    ggml_tensor * inp_out_ids = hparams.p2p_is_tail ? build_inp_out_ids() : nullptr;

    // P2P: n_layer is this device's SLICE count and model.layers is stored compactly, so the stock
    // [0, n_run) loop already computes exactly this device's slice.
    for (int il = 0; il < n_run; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            // compute Q and K and RoPE them
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        }
        if (il == n_run - 1 && hparams.p2p_is_tail && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    // BENCH: with 0 blocks run, the in-loop out_ids gather never fired, so the tail's finals would
    // run over EVERY input position. Gather here so the isolated "finals" measurement matches a real
    // forward (prefill: only the last token -> 1 lm_head row).
    if (hparams.p2p_is_tail && n_run == 0 && inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    if (!hparams.p2p_is_tail) {
        // MID / HEAD ring device: export the PRE-norm hidden state and stop. The successor device
        // consumes this as its input (via ubatch.embd). We deliberately skip output_norm + lm_head
        // (only the tail produces logits). Stored in t_h_nextn (the slot MTP/EAGLE use), which the
        // context copies out to the readable embd_nextn buffer when cparams.embeddings_nextn is set.
        cb(cur, "result_h_nextn", -1);
        res->t_h_nextn = cur;
        ggml_build_forward_expand(gf, cur);
        return;
    }

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur, model.output_s);

    if (model.output_b != nullptr) {
        cur = ggml_add(ctx0, cur, model.output_b);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
