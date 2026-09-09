"""tiered_eval.py — OFFICIAL LIBERO-Mem tiered success eval (reviewer-fixed).

Fixes that caused invalid Stage2/Stage3 comparisons:
  1. NEVER exit early on env.step()'s done: LIBERO-Mem's step can return True
     when an INTERMEDIATE subgoal is satisfied, which would truncate the
     trajectory before 3 cycles complete.
  2. Per-demo paired diffusion seed so the same demo receives the same noise
     across checkpoints (strict paired comparison).
  3. Separate reports: tiered_progress (tier/3), tier3_reached, and
     strict_success (tier==3 AND not overshot AND check_success(inc=False)).
  4. reset_subgoal_progress + _overshot=False each demo (official reset does
     not clear _overshot -> cross-episode pollution).
Protocol: 10 no-op settle, then fixed-length rollout; each physical step
advances the subgoal machine via base._check_success(inc=True) BEFORE env.step.
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
    ep_seed = args.seed + int(demo.split("_")[1])
    torch.manual_seed(ep_seed); torch.cuda.manual_seed_all(ep_seed); np.random.seed(ep_seed)
    meta=mi[TASK][demo]; instr=meta["task_description"]
    env.reset()
    base.reset_subgoal_progress()
    base._overshot=False
    obs=env.set_init_state(np.asarray(meta["initial_state"],dtype=np.float64))
    for _ in range(WAIT): obs,*_=env.step([0,0,0,0,0,0,-1.0])
    t=WAIT; q=0; max_sat=0; ep_first="True"; overshot_ever=False
    while t<args.maxsteps+WAIT:
        acts,_=vla.predict_action(image=Image.fromarray(np.ascontiguousarray(obs["agentview_image"][::-1,:])),
            instruction=instr,unnorm_key=None,use_ddim=True,num_ddim_steps=10,episode_first_frame=ep_first)
        ep_first="False"; q+=1; acts=np.asarray(acts)
        if acts.shape!=(CHUNK,7): break
        for j in range(min(args.exec,args.maxsteps+WAIT-t)):
            base._check_success(inc=True)          # advance machine
            sat=len(base._satisfied_subgoals)
            if sat>max_sat: max_sat=sat
            if base._overshot: overshot_ever=True
            a=acts[j].copy(); a[6]=-1.0 if a[6]>=0.5 else +1.0
            obs,rew,step_done,info=env.step(a.tolist())   # NEVER break on step_done
            t+=1
    final_tier=len(base._satisfied_subgoals)
    strict = bool(final_tier==3 and not base._overshot and base._check_success(inc=False))
    tier3 = bool(final_tier==3)
    rows.append(dict(demo=demo,tier=final_tier,peak=max_sat,strict=strict,tier3=tier3,
                     overshot=overshot_ever,q=q))
    print("[%s] tier=%d/3 peak=%d strict=%s tier3=%s overshot=%s q=%d"%(
        demo,final_tier,max_sat,strict,tier3,overshot_ever,q),flush=True)
env.close()
dist={c:sum(1 for r in rows if r["tier"]==c) for c in range(4)}
n_strict=sum(1 for r in rows if r["strict"])
n_t3=sum(1 for r in rows if r["tier3"])
n_ov=sum(1 for r in rows if r["overshot"])
n=len(rows)
wsum=sum(r["tier"] for r in rows)
weighted=wsum/(3.0*max(1,n))
ge1=sum(1 for r in rows if r["tier"]>=1)
ge2=sum(1 for r in rows if r["tier"]>=2)
print("\n==== TIERED EVAL ====",flush=True)
print("arm=%s exec=%d n=%d tier_dist=%s"%(arm,args.exec,n,dist),flush=True)
print("weighted_success=%.1f%% (sum_tier=%d/%.0f)  tier3_reached=%d  strict_success=%d  >=1cycle=%d  >=2cycle=%d  overshot_demos=%d"%(
      weighted*100, wsum, 3.0*n, n_t3, n_strict, ge1, ge2, n_ov),flush=True)
print("TIERED_DONE",flush=True)
