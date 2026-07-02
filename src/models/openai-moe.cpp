#include "models.h"

void llama_model_openai_moe::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);

    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    uint32_t swa_period = 2;
    ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, swa_period, false);
    hparams.set_swa_pattern(swa_period);

    hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
    hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;
    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA, hparams.rope_freq_base_train_swa, false);

    switch (hparams.n_layer()) {
        case 24: type = LLM_TYPE_20B; break;
        case 36: type = LLM_TYPE_120B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_openai_moe::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_ff_exp = hparams.n_ff_exp;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    // P2P: n_layer is this device's SLICE count (hparams was resized in load_hparams). We store
    // layers compactly at [0, n_layer) but name the GGUF tensors with the GLOBAL layer index so a
    // device holding layers [start, end) reads the right weights from the file.
    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];
        const int g = (int) hparams.p2p_global_il((uint32_t) i);   // GLOBAL GGUF layer index

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", g), {n_embd}, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", g), {n_embd}, 0);

        create_tensor_qkv(layer, g, n_embd, n_head * n_rot, n_head_kv * n_rot, n_head_kv * n_rot, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", g), {n_head * n_rot, n_embd}, 0);

        layer.attn_sinks = create_tensor(tn(LLM_TENSOR_ATTN_SINKS, "weight", g), {n_head}, 0);

        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", g), {  n_embd, n_expert}, 0);
        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", g), {  n_embd, n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", g), {n_ff_exp,   n_embd, n_expert}, 0);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", g), {  n_embd, n_ff_exp, n_expert}, 0);

        layer.wo_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", g), {n_embd}, 0);

        layer.ffn_gate_inp_b  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "bias", g), {n_expert}, 0);
        layer.ffn_gate_exps_b = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "bias", g), {n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "bias", g), {  n_embd, n_expert}, 0);
        layer.ffn_up_exps_b   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "bias", g), {n_ff_exp, n_expert}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_openai_moe::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_openai_moe::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv_iswa();

    // TAIL only: out_ids gathers the output rows before norm+lm_head. A mid/head ring device
    // exports the full-width hidden state and never gathers, so it must NOT create this input
    // (an unused input is left unallocated and would fail set_input's buffer assert).
    ggml_tensor * inp_out_ids = hparams.p2p_is_tail ? build_inp_out_ids() : nullptr;

    // P2P: n_layer is this device's SLICE count (hparams was resized in load_hparams), and
    // model.layers is stored compactly, so the stock [0, n_layer) loop already computes exactly
    // this device's slice. (Ring note: for a mid-pipeline device the input hidden state must be
    // fed into inpL from the previous device instead of build_inp_embd — wired in Phase 7.)
    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;

        const float freq_base_l  = model.get_rope_freq_base (cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, nullptr,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            // compute Q and K and RoPE them
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_rot, n_head, n_head_kv, il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, model.layers[il].attn_sinks, nullptr, 1.0f/sqrtf(float(n_rot)), il);

            cb(cur, "attn_out", il);
        }
        if (il == n_layer - 1 && hparams.p2p_is_tail) {
            // skip computing output for unused tokens — TAIL ONLY. A mid/head ring device must
            // forward the FULL-width hidden state (the next device's attention needs every
            // position), so the out_ids reduction is applied only where logits are produced.
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = ffn_inp;
        cur = build_norm(cur,
                model.layers[il].attn_post_norm, nullptr,
                LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        // MoE branch
        cur = build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp,  model.layers[il].ffn_gate_inp_b,
                model.layers[il].ffn_up_exps,   model.layers[il].ffn_up_exps_b,
                model.layers[il].ffn_gate_exps, model.layers[il].ffn_gate_exps_b,
                model.layers[il].ffn_down_exps, model.layers[il].ffn_down_exps_b,
                nullptr,
                n_expert, n_expert_used,
                LLM_FFN_SWIGLU_OAI_MOE, false,
                hparams.expert_weights_scale,
                LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT,
                il);
        cb(cur, "ffn_moe_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;   // pre-final-norm residual (the ring boundary hidden state)

    if (!hparams.p2p_is_tail) {
        // MID / HEAD ring device: export the PRE-norm hidden state and stop. The successor device
        // consumes this as its input (via ubatch.embd). We deliberately skip output_norm + lm_head
        // (only the tail produces logits). Stored in t_h_nextn (same slot MTP/EAGLE use), which the
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

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
