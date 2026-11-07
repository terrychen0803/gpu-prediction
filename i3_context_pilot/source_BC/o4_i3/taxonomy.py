"""Exact ATen overload registry. Reference/evaluator only, never predictor input."""
VERSION = 'o4_i3_taxonomy_context_BC_v2_20261107'

# Explicit names, no substring or prefix fallback. Additional overloads stay unmapped.
GROUPS = {
 'conv': ['aten.conv2d.default'],
 'batchnorm_eval': ['aten.batch_norm.default'],
 'linear': ['aten.linear.default'],
 'matmul_unresolved': ['aten.matmul.default'],
 'relu': ['aten.relu.default','aten.relu_.default'],
 'silu': ['aten.silu.default','aten.silu_.default'],
 'gelu': ['aten.gelu.default'],
 'hardsigmoid': ['aten.hardsigmoid.default'],
 'hardswish': ['aten.hardswish_.default','aten.hardswish.default'],
 'add': ['aten.add.Tensor','aten.add_.Tensor'],
 'mul': ['aten.mul.Tensor'],
 'elementwise': ['aten.sub.Tensor','aten.neg.default','aten.pow.Tensor_Scalar','aten.sigmoid.default',
                 'aten.cos.default','aten.sin.default','aten.floor_divide.default','aten.remainder.Scalar'],
 'mask_compare': ['aten.bitwise_and.Tensor','aten.eq.Scalar','aten.le.Tensor','aten.ne.Scalar',
                  'aten.masked_fill.Scalar','aten.where.ScalarOther'],
 'fill': ['aten.fill_.Tensor'],
 'maxpool': ['aten.max_pool2d.default'],
 'adaptive_average_pool': ['aten.adaptive_avg_pool2d.default'],
 'resize_bilinear': ['aten.upsample_bilinear2d.vec'],
 'resize_nearest': ['aten.upsample_nearest2d.vec'],
 'concat': ['aten.cat.default'],
 'layernorm': ['aten.layer_norm.default'],
 'softmax': ['aten.softmax.int'],
 'reduction': ['aten.max.dim','aten.mean.dim','aten.rsqrt.default'],
 'embedding_gather': ['aten.embedding.default','aten.gather.default','aten.index.Tensor'],
 'layout_copy': ['aten.contiguous.default','aten.to.device','aten.to.dtype','aten.to.dtype_layout'],
 'layout_transform': ['aten.reshape.default','aten.repeat.default','aten.pad.default','aten.roll.default'],
 'view_or_alias': ['aten.alias.default','aten.chunk.default','aten.detach.default','aten.detach_.default',
                  'aten.expand.default','aten.flatten.using_ints','aten.permute.default','aten.select.int',
                  'aten.slice.Tensor','aten.split_with_sizes.default','aten.transpose.int','aten.unsqueeze.default',
                  'aten.view.default'],
 'dropout_eval_or_unknown': ['aten.dropout.default','aten.dropout_.default'],
 'allocation_or_lift': ['aten.arange.default','aten.lift_fresh.default','aten.new_ones.default','aten.new_zeros.default'],
 'topk': ['aten.topk.default'],
 # New controlled ATen dispatch can expose these lower-level overloads.
 'gemm': ['aten.mm.default'],
 'bmm': ['aten.bmm.default'],
 'gemm_bias': ['aten.addmm.default'],
 'conv_parameters_required': ['aten.convolution.default','aten._convolution.default'],
 'batchnorm_parameters_required': ['aten.native_batch_norm.default','aten._native_batch_norm_legit_no_training.default'],
 'layernorm_native': ['aten.native_layer_norm.default'],
 'resize_bilinear_native': ['aten.upsample_bilinear2d.default'],
 'resize_nearest_native': ['aten.upsample_nearest2d.default'],
 'maxpool_indices': ['aten.max_pool2d_with_indices.default'],
}

REQUIRED = {
 'conv': ['stride','padding','dilation','groups','bias_presence'],
 'conv_parameters_required': ['stride','padding','dilation','groups','transposed','output_padding','bias_presence'],
 'batchnorm_eval': ['training','eps','momentum','running_statistics'],
 'batchnorm_parameters_required': ['eps','running_statistics'],
 'linear': ['bias_presence'], 'matmul_unresolved': ['broadcast_batch_semantics'],
 'gemm_bias': ['alpha','beta'],
 'gelu': ['approximate'], 'add': ['alpha','broadcast_semantics','inplace_alias'],
 'mul': ['broadcast_semantics'], 'elementwise': ['operator_scalar_arguments'],
 'mask_compare': ['operator_scalar_arguments','broadcast_semantics'],
 'fill': ['value'], 'maxpool': ['kernel_size','stride','padding','dilation','ceil_mode'],
 'maxpool_indices': ['kernel_size','stride','padding','dilation','ceil_mode'],
 'adaptive_average_pool': ['output_size'],
 'resize_bilinear': ['output_size','align_corners','scale_factors'],
 'resize_nearest': ['output_size','scale_factors'],
 'resize_bilinear_native': ['output_size','align_corners','scale_factors'],
 'resize_nearest_native': ['output_size','scale_factors'],
 'concat': ['dim'], 'layernorm': ['normalized_shape','eps','affine'],
 'layernorm_native': ['normalized_shape','eps','affine'], 'softmax': ['dim','dtype'],
 'reduction': ['reduction_dimensions_and_options'], 'embedding_gather': ['indexing_options'],
 'layout_copy': ['dtype_device_memory_format_options'], 'layout_transform': ['layout_parameters'],
 'view_or_alias': ['view_alias_parameters'], 'dropout_eval_or_unknown': ['p','training'],
 'allocation_or_lift': ['allocation_or_lift_parameters'], 'topk': ['k','dim','largest','sorted'],
}

BY_NAME = {}
for subtype, names in GROUPS.items():
    for name in names:
        if name in BY_NAME: raise ValueError('Duplicate overload: '+name)
        BY_NAME[name] = {'subtype': subtype, 'required_semantic_attributes': REQUIRED.get(subtype, []),
                         'common_tensor_attributes': ['shape','dtype','stride','device','layout'],
                         'label_level':'ATen_interface_not_kernel_or_DNN_layer',
                         'semantic_status':'exact_interface_name_parameters_not_yet_verified'}

def classify(name):
    return {'aten':name, 'taxonomy_version':VERSION, 'mapped':name in BY_NAME,
            **BY_NAME.get(name, {'subtype':'unmapped','required_semantic_attributes':['complete_interface_contract'],
                                 'common_tensor_attributes':['shape','dtype','stride','device','layout'],
                                 'label_level':'unmapped','semantic_status':'unverified'})}

def registry():
    return {'version':VERSION,'role':'evaluator_only','fallback':'unmapped',
            'zero_recorded_work_is_not_verified_noop':True,'entries':BY_NAME}
