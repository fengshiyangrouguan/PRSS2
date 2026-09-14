import json
r = json.load(open("/root/autodl-tmp/ret_out/joint_seed0/joint_probe.json"))
for lab in sorted(r["arms"]["Y_leaf"]):
    o = r["arms"]["Y_leaf"][lab]
    print("==", lab, "| base_probe cv=", round(o["base_probe"]["cv_oof_nll"], 5),
          "lam=", o["base_probe"]["lam"], "n_invalid=", o["base_probe"]["n_invalid_lambda"])
    for p in sorted(o["positions"], key=int):
        e = o["positions"][p]
        pr = e["probe"]
        print("   phys%s used_fb=%-5s lam=%-7s cv=%-10.5f  nll_b=%.5f nll_f=%.5f  J=%+.6f"
              % (p, pr["used_base_fallback"], pr["lam"], pr["cv_oof_nll"],
                 e["nll_base_mean"], e["nll_full_mean"], e["J_point"]))
