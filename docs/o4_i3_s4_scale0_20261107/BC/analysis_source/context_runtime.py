"""Exact R overload/argument checks for new BC node interfaces; evaluator only."""
from runtime_check_helpers import check_runtime as legacy

def check(c,op):
 f=c['family'];d=c['definition'];p=d['parameters'];args=op.get('bound_arguments',{});bad=[]
 if f in ['conv','depthwise','batchnorm_eval','relu','add','adaptive_average_pool','bilinear_resize','concat','bmm']:return legacy(c,op)
 names={'hardswish':{'aten.hardswish.default'},'hardsigmoid':{'aten.hardsigmoid.default'},'mul':{'aten.mul.Tensor'},'mul_scalar':{'aten.mul.Tensor','aten.mul.Scalar'},'softmax':{'aten.softmax.int'}}
 if op['aten'] not in names.get(f,set()):return {'matched':False,'problems':['unexpected exact overload '+op['aten']]}
 mapping={'x':'self'}
 if f=='mul':mapping['other']='other'
 for n,key in mapping.items():
  actual=args.get(key);spec=d['tensors'][n]
  if not isinstance(actual,dict):bad.append(n+':missing tensor');continue
  for k in ['shape','stride','storage_offset']:
   if actual.get(k)!=spec[k]:bad.append(n+':'+k)
  if actual.get('dtype')!='torch.float16':bad.append(n+':dtype')
 if f=='mul_scalar' and args.get('other')!=p['value']:bad.append('scalar:value')
 if f=='softmax':
  if args.get('dim')!=p['dim']:bad.append('softmax:dim')
  dtype=args.get('dtype');dtype=dtype.get('value') if isinstance(dtype,dict) else dtype
  if dtype not in ('torch.float16','float16'):bad.append('softmax:dtype:'+str(dtype))
 if op.get('output_metadata',{}).get('shape')!=d['output_shape']:bad.append('output_shape')
 return {'matched':not bad,'problems':bad,'ATen':op['aten']}
