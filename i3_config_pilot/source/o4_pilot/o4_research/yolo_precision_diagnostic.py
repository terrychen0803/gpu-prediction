"""Untimed YOLO one2one raw-head parity; discrete topk outputs kept separate."""
import argparse
from pathlib import Path
from .common import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--assets-manifest',required=True);p.add_argument('--expect-gpu',default='GB10');a=p.parse_args();out=Path(a.out);out.mkdir(exist_ok=False);gpu_guard(out,a.expect_gpu);import torch
 from .adapters import Adapter
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 with torch.inference_mode(),torch.autocast('cuda',enabled=False):
  adapter=Adapter(torch,'YO-A',a.assets_manifest);head=adapter.model.model[-1];head.export=False;cpu_y,cpu_preds=adapter.model(adapter.x);cpu={k:cpu_preds['one2one'][k].float().clone() for k in ('boxes','scores')};head.export=True;assert torch.equal(cpu_y,adapter.model(adapter.x))
  adapter.to_cuda();head=adapter.model.model[-1];head.export=False;gpu_y,gpu_preds=adapter.model(adapter.x);head.export=True;export=adapter.model(adapter.x);torch.cuda.synchronize();assert torch.equal(gpu_y,export);actual={k:gpu_preds['one2one'][k].float().cpu() for k in ('boxes','scores')};metrics={}
  for key in cpu:
   delta=actual[key]-cpu[key];l2=float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(cpu[key]).clamp_min(1e-12));scaled=float(delta.abs().max()/cpu[key].abs().max().clamp_min(1));metrics[key]={'shape':list(cpu[key].shape),'relative_l2':l2,'max_abs_over_scale':scaled,'passed_0_02_rule':l2<=.02 and scaled<=.02}
  torch.save({'CPU_FP32_one2one':cpu,'GPU_FP16_one2one':actual,'CPU_topk':cpu_y,'GPU_topk':gpu_y.float().cpu()},out/'comparison_tensors.pth')
  write(out/'SUMMARY.json',{'protocol_id':'o4_yolo_raw_head_parity_v1_20261106','weights_sha256':adapter.metadata['weights_sha256'],'input_sha256':adapter.metadata['input_sha256'],'native_export_tensor_parity_exact':True,'raw_anchor_aligned_precision':metrics,'raw_anchor_precision_passed':all(x['passed_0_02_rule'] for x in metrics.values()),'topk_rowwise_relative_l2':float(torch.linalg.vector_norm(gpu_y.float().cpu()-cpu_y)/torch.linalg.vector_norm(cpu_y).clamp_min(1e-12)),'interpretation':'raw pre-topk tensor comparison is anchor-aligned; discrete ranked output rows may change. Does not prove selected detection set equivalence or task accuracy. Original smoke correctness rule unchanged.','task_accuracy_verified':False,'strict_verified':0})
 print(read(out/'SUMMARY.json'))
if __name__=='__main__':main()
