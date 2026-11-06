import argparse,sys

def main():
 p=argparse.ArgumentParser(description='O4 GI/P/R engineering pilot; no model training or formal reference claim')
 sub=p.add_subparsers(dest='command',required=True)
 r=sub.add_parser('run');r.add_argument('--out',required=True);r.add_argument('--expect-gpu',required=True);r.add_argument('--workload',action='append');r.add_argument('--mode',action='append',choices=['GI','P','R']);r.add_argument('--repeats',type=int,default=3);r.add_argument('--trace-backend',choices=['cuda-sw','cuda'],default='cuda-sw');r.add_argument('--dry-run',action='store_true');r.add_argument('--keep-going',action='store_true')
 w=sub.add_parser('worker');w.add_argument('--request',required=True)
 a=sub.add_parser('audit');a.add_argument('--folder',required=True);a.add_argument('--out',required=True)
 s=sub.add_parser('summarize');s.add_argument('--run',required=True);s.add_argument('--out',required=True)
 args=p.parse_args()
 if args.command=='worker':
  from .worker import run
  return run(args.request)
 if args.command=='run':
  from .runner import execute
  return execute(args)
 if args.command=='audit':
  from .trace import audit
  return 1 if audit(args.folder,args.out)['status']=='failed' else 0
 from .report import summarize
 return summarize(args.run,args.out)
if __name__=='__main__':
 try:sys.exit(main())
 except Exception as exc:print(f'{type(exc).__name__}: {exc}',file=sys.stderr);sys.exit(1)
