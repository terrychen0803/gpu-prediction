"""Reference-only complete call arguments without tensor values or device sync."""
def describe(value,torch):
    if isinstance(value,torch.Tensor):
        return {'kind':'tensor','shape':list(value.shape),'stride':list(value.stride()),'dtype':str(value.dtype),
                'device':str(value.device),'layout':str(value.layout),'storage_offset':value.storage_offset(),
                'requires_grad':value.requires_grad}
    if isinstance(value,(tuple,list)):return [describe(x,torch) for x in value]
    if isinstance(value,dict):return {str(k):describe(v,torch) for k,v in value.items()}
    if value is None or isinstance(value,(str,bool,int,float)):return value
    return {'kind':'typed_value','type':type(value).__name__,'value':str(value)}

def arguments(func,args,kwargs,torch):
    result={}
    for i,arg in enumerate(func._schema.arguments):
        if i<len(args):value=args[i]
        elif arg.name in kwargs:value=kwargs[arg.name]
        elif arg.has_default_value():value=arg.default_value
        else:raise ValueError('Unbound schema argument '+arg.name)
        result[arg.name]=describe(value,torch)
    return result

def inventory(torch):
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten
    class Inventory(TorchDispatchMode):
        def __init__(self):super().__init__();self.rows=[];self.active=False
        def begin(self,stage,iteration):self.active=True;self.stage=stage;self.iteration=iteration;self.index=0
        def end(self):self.active=False
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            kwargs=kwargs or {}
            if not self.active:return func(*args,**kwargs)
            label=f'o4.op/{self.stage}/{self.iteration:06d}/{self.index:04d}';self.index+=1
            ts=[x for x in tree_flatten((args,kwargs))[0] if isinstance(x,torch.Tensor)]
            row={'scope':label,'stage':self.stage,'iteration':self.iteration,'op_index':self.index-1,
                 'aten':str(func),'input_shapes':[list(t.shape) for t in ts],
                 'input_dtypes':[str(t.dtype) for t in ts],'argument_schema':str(func._schema),
                 'bound_arguments':arguments(func,args,kwargs,torch),'metadata_role':'R_only_never_predictor'}
            torch.cuda.nvtx.range_push(label)
            try:output=func(*args,**kwargs)
            finally:torch.cuda.nvtx.range_pop()
            row['output_metadata']=describe(output,torch);self.rows.append(row);return output
    return Inventory()
