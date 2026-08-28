
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/clip_by_value_tiling.h"
#include "../op_kernel/tiling_key_clip_by_value.h"

namespace optiling {
    
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t core_num = platform.GetCoreNumAiv();
        uint64_t ub_total_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_total_size);

        // 璇诲彇绠楀瓙涓変釜杈撳叆寮犻噺
        const gert::Tensor *input_x = context->GetRequiredInputTensor(0);
        const gert::Tensor *input_min = context->GetRequiredInputTensor(1);
        const gert::Tensor *input_max = context->GetRequiredInputTensor(2);

        // 鑾峰彇杈撳叆鏁版嵁绫诲瀷銆佸崟鍏冪礌瀛楄妭銆佹€诲厓绱犳暟閲?
        ge::DataType data_type = input_x->GetDataType();
        int single_dtype_bytes = ge::GetSizeByDataType(data_type);
        uint32_t total_x_elem = input_x->GetShapeSize();
        uint32_t min_elem_cnt = input_min->GetShapeSize();
        uint32_t max_elem_cnt = input_max->GetShapeSize();

        // 鍒ゆ柇min/max鏄惁涓烘爣閲忥紙鍏冪礌鏁?=1锛?
        bool min_is_scalar = (min_elem_cnt <= 1);
        bool max_is_scalar = (max_elem_cnt <= 1);

        // 缁戝畾Tiling妯℃澘鍙傛暟锛屽尯鍒嗕笉鍚屾暟鎹被鍨媖ernel鍒嗘敮
        uint32_t dtype_tag = static_cast<uint32_t>(data_type);
        ASCENDC_TPL_SEL_PARAM(context, dtype_tag);

        // 纭欢鍚戦噺鍧楀浐瀹?2瀛楄妭瀵归綈
        constexpr uint32_t VEC_BLOCK_BYTE = 32;
        uint32_t vec_per_block = VEC_BLOCK_BYTE / static_cast<uint32_t>(single_dtype_bytes);
        uint32_t use_core_cnt = static_cast<uint32_t>(core_num);
        if (use_core_cnt == 0) use_core_cnt = 1;

        // 鏁版嵁閲忚繃灏忓垯寮哄埗鍗曟牳杩愯
        if (total_x_elem < vec_per_block && use_core_cnt > 1) {
            use_core_cnt = 1;
        }

        // 鏍规嵁min/max鏄惁鏍囬噺璁＄畻UB闇€瑕佺殑缂撳瓨鏁伴噺
        uint32_t buffer_amount;
        if (min_is_scalar && max_is_scalar) {
            buffer_amount = 4;
        } else if (min_is_scalar || max_is_scalar) {
            buffer_amount = 6;
        } else {
            buffer_amount = 8;
        }

        // 璁＄畻鍗晅ile鏈€澶у彲瀹圭撼鍏冪礌锛屾寜鍚戦噺鍧楀榻?
        uint32_t max_tile_elem = static_cast<uint32_t>(ub_total_size / (buffer_amount * static_cast<uint64_t>(single_dtype_bytes)));
        uint32_t tile_unit_len = max_tile_elem / vec_per_block * vec_per_block;
        if (tile_unit_len < vec_per_block) {
            tile_unit_len = vec_per_block;
        }

        // 鍗曟牳鍒嗛厤鎬婚暱搴︼紝鍚戜笂瀵归綈tile闀垮害
        uint32_t rough_core_data = (total_x_elem + use_core_cnt - 1) / use_core_cnt;
        uint32_t single_core_len = ((rough_core_data + tile_unit_len - 1) / tile_unit_len) * tile_unit_len;
        if (single_core_len == 0) single_core_len = tile_unit_len;

        // 瀹為檯鍚敤鐨勬牳蹇冩暟閲?
        uint32_t real_core_num = (total_x_elem + single_core_len - 1) / single_core_len;
        if (real_core_num == 0) real_core_num = 1;

        // 闄愬埗鍗晅ile闀垮害涓嶈秴杩囧崟鏍告€绘暟鎹?
        if (tile_unit_len > single_core_len) {
            tile_unit_len = single_core_len;
        }

        // 濉厖Tiling缁撴瀯浣擄紝涓嬪彂鑷矺ernel渚?
        ClipByValueTilingData *tile_param = context->GetTilingData<ClipByValueTilingData>();
        tile_param->length = total_x_elem;
        tile_param->blockLength = single_core_len;
        tile_param->tileLength = tile_unit_len;
        tile_param->isScalarMin = min_is_scalar ? 1 : 0;
        tile_param->isScalarMax = max_is_scalar ? 1 : 0;

        // 璁剧疆骞惰鏍告暟锛屾棤闇€workspace
        context->SetBlockDim(real_core_num);
        size_t *workspace_buf = context->GetWorkspaceSizes(1);
        workspace_buf[0] = 0;

        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    // 杈撳嚭Shape鎺ㄥ锛氳緭鍑簊hape涓庤緭鍏瀹屽叏涓€鑷?
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *x_shape = context->GetInputShape(0);
        gert::Shape *y_shape = context->GetOutputShape(0);
        if (x_shape == nullptr || y_shape == nullptr) {
            return GRAPH_FAILED;
        }
        *y_shape = *x_shape;
        return GRAPH_SUCCESS;
    }

    // 杈撳嚭鏁版嵁绫诲瀷鎺ㄥ锛氳緭鍑虹被鍨嬭窡闅忚緭鍏
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        ge::DataType input_dtype = context->GetInputDataType(0);
        context->SetOutputDataType(0, input_dtype);
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    // ClipByValue绠楀瓙娉ㄥ唽绫伙紝瀹氫箟杈撳叆杈撳嚭銆佺被鍨嬨€乀iling缁戝畾
    class ClipByValue : public OpDef {
    public:
        explicit ClipByValue(const char *name) : OpDef(name) {
            // 杈撳叆x瀹氫箟锛屾敮鎸乫p16/fp32/int32 ND鏍煎紡
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            // 涓嬮檺杈撳叆
            this->Input("clip_value_min")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            // 涓婇檺杈撳叆
            this->Input("clip_value_max")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            // 瑁佸壀缁撴灉杈撳嚭
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

            // 缁戝畾shape涓巇type鎺ㄥ鍑芥暟
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

            // 缁戝畾AICore Tiling鍒嗗潡鍑芥暟锛岄€傞厤910B鑺墖
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(ClipByValue);
}  // namespace ops