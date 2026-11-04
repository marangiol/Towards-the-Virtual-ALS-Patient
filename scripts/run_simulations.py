from pathlib import Path
import sys,argparse
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.simulation import make_design,run_bank
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/demo.json')
    p.add_argument('--mode',choices=['uniform','regional'],default='uniform')
    p.add_argument('--count',type=int,default=100)
    p.add_argument('--grid',action='store_true')
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    run_bank(make_design(args.mode,args.count,args.grid),ROOT/args.config,ROOT/args.output,args.workers)
