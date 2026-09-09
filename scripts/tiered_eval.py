"""tiered_eval.py — OFFICIAL LIBERO-Mem tiered success eval.

Fixes the evaluator bug: official success requires advancing the subgoal
state machine via env._check_success(inc=True) EVERY physical step, plus
reading len(_satisfied_subgoals) as tiered progress.  Our earlier evals never
called inc=True, so _satisfied_subgoals stayed empty -> always 0/3 even when
the model actually completed pickups (the user's video showed 2).

Protocol:
  * 10 no-op settle
  * reset_subgoal_progress + _overshot=False before each demo
  * per executed env step: base._check_success(inc=True) THEN env.step
  * after each step read satisfied = len(base._satisfied_subgoals)
  * report tiered_success distribution {0,1,2,3}, final check_success
"""
import os, sys, json, argparse
import numpy as np
os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com"); os.environ.setdefault("HF_HUB_OFFLINE","1")
os.environ.setdefault("LLAMA2_LOCAL_PATH","/root/autodl-tmp/Llama-2-7b-hf")
sys.path.insert(0,"/root/autodl-tmp/libero-mem-code"); sys.path.insert(0,"/root/autodl-tmp/vla")
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from libero.libero.envs import OffScreenRenderEnv
from vla import load_vla
TASK="KITCHEN_SCENE1_3_lift_the_bowl_and_place_it_back_on_the_plate_3_times"
BASE="/root/autodl-tmp/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt"
BDDL="/root/autodl-tmp/libero-mem-code/libero/libero/bddl_files/libero_mem/%s.bddl"%TASK
MI="/root/autodl-tmp/libero-mem/metainfo.json"
WAIT=10; CHUNK=16
ap=argparse.ArgumentParser(); ap.add_argument("--ckpt",required=True)
ap.add_argument("--demo-start",type=int,default=1); ap.add_argument("--n",type=int,default=20)
ap.add_argument("--exec",type=int,default=8); ap.add_argument("--maxsteps",type=int,default=370)
ap.add_argument("--seed",type=int,default=42)
args=ap.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed)
ck=torch.load(args.ckpt,map_location="cpu",weights_only=False)
arm,mem=ck["arm"],ck.get("mem_length",16)
vla=load_vla(model_id_or_path=BASE,hf_token=None,load_for_training=True,use_bf16=True,
  action_dim=7,future_action_window_size=15,action_model_type="DiT-L",use_ema=False,
  dataloader_type="stream",mem_length=mem,retrieval_layers=2,use_timestep_pe=True,
  fusion_type="gate",consolidate_type="tome",update_fused=False,per_token_size=256,
  use_rpbe_gamma=(arm in ("gamma-task","gamma-rpbe")),gamma_rank=64,gamma_alpha_init=1.0,
  rpbe_merge_records=(arm!="avg"),rpbe_task_grad=(arm in ("gamma-task","gamma-rpbe")),rpbe_seed=42)
vla.vlm.requires_grad_(False)
lc=ck["lora_config"]
vla.vlm.llm_backbone.llm=get_peft_model(vla.vlm.llm_backbone.llm,LoraConfig(
  r=lc["r"],lora_alpha=lc["lora_alpha"],lora_dropout=lc["lora_dropout"],
  target_modules="all-linear",task_type="CAUSAL_LM"))
named=dict(vla.named_parameters())
for n,t in ck["model"].items():
    if n in named and named[n].requires_grad: named[n].data.copy_(t.to(named[n].dtype))
with open(os.path.join(os.path.dirname(args.ckpt),"dataset_statistics.json")) as f:
    vla.norm_stats=json.load(f)
vla.eval()
mi=json.load(open(MI))
env=OffScreenRenderEnv(bddl_file_name=BDDL,camera_heights=256,camera_widths=256)
base=env.env if hasattr(env,"env") else env
rows=[]
for i in range(args.n):
    demo="demo_%d"%(args.demo_start+i)
    if demo not in mi[TASK]: continue
    meta=mi[TASK][demo]; instr=meta["task_description"]
    env.reset()
    base.reset_subgoal_progress()
    base._overshot=False          # official reset bug: overshot not cleared
    obs=env.set_init_state(np.asarray(meta["initial_state"],dtype=np.float64))
    for _ in range(WAIT): obs,*_=env.step([0,0,0,0,0,0,-1.0])
    t=WAIT; q=0; ep_first="True"; done=False; max_sat=0
    peak_tier={}; tier_step={}
    while t<args.maxsteps+WAIT:
        acts,_=vla.predict_action(image=Image.fromarray(np.ascontiguousarray(obs["agentview_image"][::-1,:])),
            instruction=instr,unnorm_key=None,use_ddim=True,num_ddim_steps=10,episode_first_frame=ep_first)
        ep_first="False"; q+=1; acts=np.asarray(acts)
        if acts.shape!=(CHUNK,7): break
        for j in range(min(args.exec,args.maxsteps+WAIT-t)):
            # OFFICIAL: advance subgoal state machine every physical step
            base._check_success(inc=True)
            sat=len(base._satisfied_subgoals)
            if sat>max_sat: max_sat=sat; tier_step[sat]=t-WAIT
            a=acts[j].copy(); a[6]=-1.0 if a[6]>=0.5 else +1.0
            obs,rew,done,info=env.step(a.tolist()); t+=1
            if done and done is not None: break
        if done: break
    final_sat=len(base._satisfied_subgoals)
    try: fin=bool(base._check_success(inc=False))
    except Exception: fin=bool(done)
    rows.append(dict(demo=demo,tier=final_sat,peak=max_sat,fin=fin,done=bool(done),q=q))
    print("[%s] tier=%d/3 peak=%d done=%s fin=%s q=%d"%(demo,final_sat,max_sat,done,fin,q),flush=True)
env.close()
dist={c:sum(1 for r in rows if r["tier"]==c) for c in range(4)}
nf=sum(1 for r in rows if r["fin"])
print("\n==== TIERED EVAL ====",flush=True)
n=len(rows)
wsum=sum(r["tier"] for r in rows)
weighted=wsum/(3.0*max(1,n))
full=sum(1 for r in rows if r["tier"]==3)
ge1=sum(1 for r in rows if r["tier"]>=1)
ge2=sum(1 for r in rows if r["tier"]>=2)
print("arm=%s exec=%d n=%d tier_dist=%s final_success=%d"%(arm,args.exec,n,dist,nf),flush=True)
print("weighted_success=%.1f%% (sum_tier=%d/%.0f)  full3/3=%d  >=1cycle=%d  >=2cycle=%d"%
      (weighted*100, wsum, 3.0*n, full, ge1, ge2),flush=True)
print("TIERED_DONE",flush=True)
