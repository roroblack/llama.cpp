#ifndef HTP_OPNODE_H
#define HTP_OPNODE_H

#define GGML_COMMON_IMPL_CPP
#include "ggml-backend-impl.h"
#include "ggml-common.h"

#include <algorithm>
#include <string>
#include <vector>
#include <memory>
#include <stdio.h>
#include "htp-ops.h"
#include "htp/matmul-ops.h"
#include "htp/flash-attn-ops.h"
#include "htp/unary-ops.h"
#include "htp/allreduce-ops.h"

struct htp_opnode {
    ggml_tensor * node   { nullptr };
    htp_op_code   opcode { HTP_OP_INVALID };
    int32_t       kernel_params[HTP_OP_MAX_KERN_PARAMS] {0};

    std::vector<ggml_tensor *>                fused;
    std::vector<std::shared_ptr<ggml_tensor>> dummy;

    std::vector<const ggml_tensor *> inputs;
    std::vector<const ggml_tensor *> outputs;
    std::string                      name;

    // n-chunking: this node computes rows [chunk_row_first, +ne[1]) of a larger
    // MUL_MAT whose full activation block does not fit VTCM. chunk_src1/chunk_dst
    // point into `dummy` and their data pointers are re-derived from the parents
    // at enqueue time, because a cached graph replays the same opnodes while the
    // allocator may have moved the parent tensors underneath.
    bool                chunked           { false };
    int64_t             chunk_row_first   { 0 };
    ggml_tensor *       chunk_src1        { nullptr };
    ggml_tensor *       chunk_dst         { nullptr };
    const ggml_tensor * chunk_parent_src1 { nullptr };
    const ggml_tensor * chunk_parent_dst  { nullptr };

    void refresh_chunk() {
        if (!chunked) {
            return;
        }
        // Refresh the allocation metadata too, not just the address. add_tensor() computes
        // the DSP-side offset as (data - buffer->base()) and reads ->extra, so a chunk view
        // carrying a new address next to a stale buffer would hand the DSP a bogus offset.
        refresh_chunk_view(chunk_src1, chunk_parent_src1);
        refresh_chunk_view(chunk_dst,  chunk_parent_dst);
    }

    void refresh_chunk_view(ggml_tensor * view, const ggml_tensor * parent) {
        view->buffer = parent->buffer;
        view->extra  = parent->extra;
        view->type   = parent->type;
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            view->nb[i] = parent->nb[i];
        }
        view->ne[0] = parent->ne[0];
        view->ne[2] = parent->ne[2];
        view->ne[3] = parent->ne[3];
        view->data  = (char *) parent->data + chunk_row_first * parent->nb[1];
    }

    int n_active_src(const ggml_tensor * t) const {
        if (!t) return 0;
        for (int i = GGML_MAX_SRC - 1; i >= 0; i--) {
            if (t->src[i]) {
                return i + 1;
            }
        }
        return 0;
    }

    void init(ggml_tensor * node) {
        this->node = node;
        if (this->node) {
            this->name = ggml_op_desc(this->node);

            // Build inputs (preserving optional nullptrs)
            int n_inputs = n_active_src(this->node);
            this->inputs.resize(n_inputs, nullptr);
            for (int i = 0; i < n_inputs; i++) {
                this->inputs[i] = this->node->src[i];
            }

            // Build outputs
            this->outputs.push_back(this->dst());
        }
    }

    htp_opnode(htp_op_code opcode = HTP_OP_INVALID, ggml_tensor * node = nullptr) : opcode(opcode) {
        init(node);
    }

    ggml_op             op()   const { return node->op; }
    const ggml_tensor * src0() const { return node->src[0]; }
    const ggml_tensor * src1() const { return node->src[1]; }
    const ggml_tensor * dst()  const { return outputs.empty() ? node : outputs.back(); }

    ggml_tensor * add_dummy(const ggml_tensor & t) {
        dummy.push_back(std::make_shared<ggml_tensor>(t));
        return dummy.back().get();
    }

    void add_fused(ggml_tensor * t, bool extra_dst = false) {
        fused.push_back(t);

        name += "+";
        name += ggml_op_desc(t);

        if (extra_dst) {
            outputs.push_back(t);
        } else {
            outputs.clear();
            outputs.push_back(t);
        }

        // Remove the newly fused intermediate output tensor t from inputs (if it was there)
        inputs.erase(std::remove(inputs.begin(), inputs.end(), t), inputs.end());

        // Append new inputs from t, preserving middle nullptrs
        int n_inputs = n_active_src(t);
        for (int i = 0; i < n_inputs; i++) {
            const auto * src = t->src[i];
            if (!src) {
                inputs.push_back(nullptr);
            } else if (src != node &&
                       std::find(fused.begin(), fused.end(), src) == fused.end() &&
                       std::find(inputs.begin(), inputs.end(), src) == inputs.end()) {
                inputs.push_back(src);
            }
        }
    }

    const std::vector<const ggml_tensor *> & get_inputs() const {
        return inputs;
    }

    const std::vector<const ggml_tensor *> & get_outputs() const {
        return outputs;
    }

    std::string op_name() const {
        return name;
    }

    bool is_empty() const {
        return ggml_op_is_empty(node->op);
    }

    bool stackable() const {
        switch (this->op()) {
            case GGML_OP_MUL_MAT:
            case GGML_OP_MUL_MAT_ID:
                return ggml_is_quantized(this->src0()->type);
            default:
                return false;
        }
    }

    bool same_input(const htp_opnode& n) const {
        return n.src1() == this->src1();
    }
};

struct htp_opformat {
    char strides[64 * GGML_MAX_SRC];
    char dims[64 * GGML_MAX_SRC];
    char types[16 * GGML_MAX_SRC];
    char buffs[64 * GGML_MAX_SRC];
    char names[64 * GGML_MAX_SRC];
    char kparams[128];

    int format_tensor_dims(char * str, size_t max_size, const struct ggml_tensor * t) {
        if (!t) {
            return snprintf(str, max_size, "NONE");
        }
        if (t->ne[2] == 1 && t->ne[3] == 1) {
            return snprintf(str, max_size, "%d:%d", (int) t->ne[0], (int) t->ne[1]);
        } else {
            return snprintf(str, max_size, "%d:%d:%d:%d", (int) t->ne[0], (int) t->ne[1], (int) t->ne[2], (int) t->ne[3]);
        }
    }

    void format_op_dims(char * str, size_t max_size, const htp_opnode & node) {
        char * p = str;
        char * p_end = str + max_size;
        auto inputs = node.get_inputs();

        if (!inputs.empty()) {
            p += std::min((size_t)format_tensor_dims(p, p_end - p, inputs[0]), (size_t)(p_end - p));

            for (size_t i = 1; i < inputs.size(); i++) {
                if (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                if (p < p_end) {
                    p += std::min((size_t)format_tensor_dims(p, p_end - p, inputs[i]), (size_t)(p_end - p));
                }
            }

            if (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        char self[64];
        format_tensor_dims(self, sizeof(self), node.dst());
        if (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", self), (size_t)(p_end - p));
        }
    }

    int format_tensor_strides(char * str, size_t max_size, const struct ggml_tensor * t) {
        if (!t) {
            return snprintf(str, max_size, "NONE");
        }
        const char * c = ggml_is_contiguous(t) ? "" : "!";

        if (t->ne[2] == 1 && t->ne[3] == 1) {
            return snprintf(str, max_size, "%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], c);
        } else {
            return snprintf(str, max_size, "%zu:%zu:%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], (size_t) t->nb[2], (size_t) t->nb[3], c);
        }
    }

    void format_op_strides(char * str, size_t max_size, const htp_opnode & node) {
        char * p = str;
        char * p_end = str + max_size;
        auto inputs = node.get_inputs();

        if (!inputs.empty()) {
            p += std::min((size_t)format_tensor_strides(p, p_end - p, inputs[0]), (size_t)(p_end - p));

            for (size_t i = 1; i < inputs.size(); i++) {
                if (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                if (p < p_end) {
                    p += std::min((size_t)format_tensor_strides(p, p_end - p, inputs[i]), (size_t)(p_end - p));
                }
            }

            if (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        char self[64];
        format_tensor_strides(self, sizeof(self), node.dst());
        if (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", self), (size_t)(p_end - p));
        }
    }

    void format_op_types(char * str, size_t max_size, const htp_opnode & node) {
        char * p = str;
        char * p_end = str + max_size;
        auto inputs = node.get_inputs();

        if (!inputs.empty()) {
            if (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[0] ? ggml_type_name(inputs[0]->type) : "NONE"), (size_t)(p_end - p));
            }

            for (size_t i = 1; i < inputs.size(); i++) {
                if (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                if (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[i] ? ggml_type_name(inputs[i]->type) : "NONE"), (size_t)(p_end - p));
                }
            }

            if (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        if (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", ggml_type_name(node.dst()->type)), (size_t)(p_end - p));
        }
    }

    const char * tensor_buff_name(const struct ggml_tensor * t) {
        if (t && t->buffer) {
            return ggml_backend_buffer_name(t->buffer);
        }
        return "NONE";
    }

    void format_op_buffs(char * str, size_t max_size, const htp_opnode & node) {
        char * p = str;
        char * p_end = str + max_size;
        auto inputs = node.get_inputs();

        if (!inputs.empty()) {
            if (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(inputs[0])), (size_t)(p_end - p));
            }

            for (size_t i = 1; i < inputs.size(); i++) {
                if (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                if (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(inputs[i])), (size_t)(p_end - p));
                }
            }

            if (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        if (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(node.dst())), (size_t)(p_end - p));
        }
    }

    void format_op_names(char * str, size_t max_size, const htp_opnode & node) {
        char * p = str;
        char * p_end = str + max_size;
        auto inputs = node.get_inputs();

        if (!inputs.empty()) {
            if (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[0] ? inputs[0]->name : "NONE"), (size_t)(p_end - p));
            }

            for (size_t i = 1; i < inputs.size(); i++) {
                if (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                if (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[i] ? inputs[i]->name : "NONE"), (size_t)(p_end - p));
                }
            }

            if (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        if (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", node.dst()->name), (size_t)(p_end - p));
        }
    }
    void format_kernel_params(char * str, size_t max_size, const htp_opnode & node) {
        if (node.opcode == HTP_OP_MUL_MAT || node.opcode == HTP_OP_MUL_MAT_ID ||
            node.opcode == HTP_OP_MUL_MAT_NX || node.opcode == HTP_OP_MUL_MAT_ID_NX ||
            node.opcode == HTP_OP_MUL_MAT_ADD) {
            const auto * kparams = (const struct htp_mm_kernel_params *) node.kernel_params;
            const char * path = "unknown";
            int32_t type = kparams->kernel_type;
            if (type == HTP_MM_KERNEL_HMX_2D || type == HTP_MM_KERNEL_HMX_F16_BATCHED) {
                path = "hmx-tiled";
            } else if (type == HTP_MM_KERNEL_HVX_F16_F16_VTCM || type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
                       type == HTP_MM_KERNEL_HVX_QUANT_ROW    || type == HTP_MM_KERNEL_HVX_QUANT_BLOCK) {
                path = "hvx-tiled";
            } else if (type == HTP_MM_KERNEL_HVX_F16_F16_DDR  || type == HTP_MM_KERNEL_HVX_F16_F32_DDR ||
                       type == HTP_MM_KERNEL_HVX_F32_F32_DDR  || type == HTP_MM_KERNEL_HVX_F32_F16_DDR ||
                       type == HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT) {
                path = "hvx-flat";
            } else if (type == HTP_MM_KERNEL_HMX_Q4_INT) {
                path = "hmx-int";
            }
            snprintf(str, max_size, "%s vtcm %d", path, (int) kparams->vtcm_size);
        } else if (node.opcode == HTP_OP_FLASH_ATTN_EXT) {
            const auto * kparams = (const struct htp_fa_kernel_params *) node.kernel_params;
            const char * path = "unknown";
            int32_t type = kparams->kernel_type;
            if (type == HTP_FA_KERNEL_HMX) {
                path = kparams->u.hmx.pipeline ? "hmx-pipe" : "hmx-seq";
            } else if (type == HTP_FA_KERNEL_HVX) {
                path = "hvx";
            }
            snprintf(str, max_size, "%s vtcm %d", path, (int) kparams->vtcm_size);
        } else if (htp_op_is_unary(node.opcode)) {
            const auto * kparams = (const struct htp_unary_kernel_params *) node.kernel_params;
            snprintf(str, max_size, "%s vtcm %d", kparams->col_tile ? "wide-row" : "row-block", (int) kparams->vtcm_size);
        } else {
            snprintf(str, max_size, "----");
        }
    }

    void format(const htp_opnode & node) {
        format_op_dims(dims, sizeof(dims), node);
        format_op_strides(strides, sizeof(strides), node);
        format_op_types(types, sizeof(types), node);
        format_op_buffs(buffs, sizeof(buffs), node);
        format_op_names(names, sizeof(names), node);
        format_kernel_params(kparams, sizeof(kparams), node);
    }

    htp_opformat() {
        strides[0] = '\0';
        dims[0]    = '\0';
        types[0]   = '\0';
        buffs[0]   = '\0';
        names[0]   = '\0';
        kparams[0] = '\0';
    }
    htp_opformat(const htp_opnode & node) { format(node); }
};

#endif // HTP_OPNODE_H
