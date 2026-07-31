#include "models.h"

#include <algorithm>   // std::min (BENCH: n_run cap)

void llama_model_llama::load_arch_hparams(llama_model_loader & ml) {
    uint32_t n_vocab = 0;
    ml.get_key(LLM_KV_VOCAB_SIZE, n_vocab, false) || ml.get_arr_n(LLM_KV_TOKENIZER_LIST, n_vocab, false);

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    if (hparams.n_expert == 8) {
        switch (hparams.n_layer()) {
            case 32: type = LLM_TYPE_8x7B; break;
            case 56: type = LLM_TYPE_8x22B; break;
            default: type = LLM_TYPE_UNKNOWN;
        }
    } else {
        switch (hparams.n_layer()) {
            case 16: type = LLM_TYPE_1B; break; // Llama 3.2 1B
            case 22: type = LLM_TYPE_1B; break;
            case 26: type = LLM_TYPE_3B; break;
            case 28: type = LLM_TYPE_3B; break; // Llama 3.2 3B
            case 30: type = LLM_TYPE_256M; break; // smoldocling 256M
            // granite uses a vocab with len 49152
            case 32: type = n_vocab == 49152 ? LLM_TYPE_3B : (n_vocab < 40000 ? LLM_TYPE_7B : LLM_TYPE_8B); break;
            case 36: type = LLM_TYPE_8B; break; // granite
            case 40: type = LLM_TYPE_13B; break;
            case 48: type = LLM_TYPE_34B; break;
            case 60: type = LLM_TYPE_30B; break;
            case 80: type = hparams.n_head() == hparams.n_head_kv() ? LLM_TYPE_65B : LLM_TYPE_70B; break;
            default: type = LLM_TYPE_UNKNOWN;
        }
    }
}

void llama_model_llama::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

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

        create_tensor_qkv(layer, g, n_embd, n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", g), {n_embd_head_k * n_head, n_embd}, 0);

        // optional bias tensors
        layer.wo_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", g), {n_embd}, TENSOR_NOT_REQUIRED);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", g), {n_embd}, 0);

        // P2P LIMITATION -- rope factors and layer slicing do not compose.
        //
        // These are MODEL-level tensors stored once at blk.0 and aliased onto every layer
        // (LLM_TENSOR_LAYER_REPEATING + TENSOR_DUPLICATED). create_tensor() maps a global bid back to
        // this device's compact slot, so a slice [30,60) can only ever address blk.30..blk.59 -- it
        // CANNOT reach blk.0. (Passing a literal 0 underflows: 0 - 30 wraps to 1.8e19 and trips
        // vector::_M_range_check inside the loader. Found by p2p_ring_boundary, 2026-07-31.)
        //
        // Consequence: on a non-head slice these resolve to nullptr and that device runs UNSCALED rope
        // while the head has it -- silently divergent, no error. So refuse loudly for LongRoPE, where
        // the factors are required for correctness. The plain rope_freqs path stays permissive because
        // the tensor is genuinely absent from most models (verified: Llama-3-14B-Instruct-v1 has ZERO
        // rope tensors of any kind), and we cannot distinguish "absent from the GGUF" from "absent
        // because sliced" at this point.
        if (hparams.rope_scaling_type_train == LLAMA_ROPE_SCALING_TYPE_LONGROPE) {
            if (hparams.p2p_partitioned() && hparams.p2p_layer_start > 0) {
                throw std::runtime_error("P2P: LongRoPE factors live at blk.0 and are unreachable from a "
                                         "slice that does not start at layer 0; this model cannot be "
                                         "partitioned with the current loader");
            }
            layer.rope_long  = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_LONG,  "weight", g), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
            layer.rope_short = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_SHORT, "weight", g), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
        }
        else {
            layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", g), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
        }

        if (n_expert == 0) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", g), {n_embd,   n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", g), {  n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", g), {n_embd,   n_ff}, 0);

            // optional MLP bias
            layer.ffn_gate_b = create_tensor(tn(LLM_TENSOR_FFN_GATE, "bias", g), {n_ff}, TENSOR_NOT_REQUIRED);
            layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", g), {n_embd}, TENSOR_NOT_REQUIRED);
            layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias", g), {n_ff}, TENSOR_NOT_REQUIRED);
        } else {
            layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", g), {n_embd, n_expert}, 0);
            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", g), {n_embd,   n_ff, n_expert}, TENSOR_NOT_REQUIRED);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", g), {  n_ff, n_embd, n_expert}, 0);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", g), {n_embd,   n_ff, n_expert}, 0);

            // For Granite MoE Shared
            if (hparams.n_ff_shexp > 0) {
                layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", g), {n_embd, hparams.n_ff_shexp}, 0);
                layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", g), {n_embd, hparams.n_ff_shexp}, 0);
                layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", g), {hparams.n_ff_shexp, n_embd}, 0);
            }
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_llama::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph<false>>(*this, params);
}

template <bool embed>
llama_model_llama::graph<embed>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
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

    using inp_attn_type = std::conditional_t<embed, llm_graph_input_attn_no_cache, llm_graph_input_attn_kv>;

    inp_attn_type * inp_attn = nullptr;
    if (n_run > 0) {
        if constexpr (embed) {
            inp_attn = build_attn_inp_no_cache();
        } else {
            inp_attn = build_attn_inp_kv();
        }
    }

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    // TAIL only: out_ids gathers the output rows before norm+lm_head. A mid/head ring device
    // exports the full-width hidden state and never gathers, so it must NOT create this input.
    ggml_tensor * inp_out_ids = hparams.p2p_is_tail ? build_inp_out_ids() : nullptr;

    // P2P: n_layer is this device's SLICE count and model.layers is stored compactly, so the stock
    // [0, n_run) loop already computes exactly this device's slice.
    for (int il = 0; il < n_run; ++il) {
        res->t_layer_inp[il] = inpL;

        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            // rope freq factors for llama3; may return nullptr for llama2 and other models
            ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

            // compute Q and K and RoPE them
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            if (hparams.use_kq_norm) {
                // Llama4TextL2Norm
                Qcur = ggml_rms_norm(ctx0, Qcur, hparams.f_norm_rms_eps);
                Kcur = ggml_rms_norm(ctx0, Kcur, hparams.f_norm_rms_eps);
                cb(Qcur, "Qcur_normed", il);
                cb(Kcur, "Kcur_normed", il);
            }
            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);
        }
        if (il == n_run - 1 && hparams.p2p_is_tail && inp_out_ids) {
            // skip computing output for unused tokens - TAIL ONLY. A mid/head ring device must
            // forward the FULL-width hidden state (the next device's attention needs every
            // position), so the out_ids reduction is applied only where logits are produced.
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network (non-MoE)
        if (model.layers[il].ffn_gate_inp == nullptr) {

            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   model.layers[il].ffn_up_s,
                    model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, model.layers[il].ffn_gate_s,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, model.layers[il].ffn_down_s,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            // MoE branch
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true,
                    hparams.expert_weights_scale,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il,
                    nullptr, nullptr,
                    model.layers[il].ffn_up_exps_s,
                    model.layers[il].ffn_gate_exps_s,
                    model.layers[il].ffn_down_exps_s);
            cb(cur, "ffn_moe_out", il);
        }
        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "ffn_out", il);

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

    if constexpr (!embed) {
        // lm_head
        cur = build_lora_mm(model.output, cur, model.output_s);

        cb(cur, "result_output", -1);
        res->t_logits = cur;
    }

    ggml_build_forward_expand(gf, cur);
}

template struct llama_model_llama::graph<false>;
template struct llama_model_llama::graph<true>;
