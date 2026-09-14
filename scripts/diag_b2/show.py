import json
r = json.load(open("/root/autodl-tmp/ret_out/joint_seed0/joint_probe.json"))
print("lams", r["lams"], "n_boot", r["n_boot"], "n_blocks", r["n_blocks"])
for line in ("Y_leaf", "Y_a2", "Y_a1"):
    arms = r["arms"].get(line, {})
    print()
    print("#### line", line)
    for lab, o in arms.items():
        print("  %-34s RootGain=%9.3f mbit  PRI=%+.4f  J_src=%+.4f  folds=%d  pairs=%d/%d"
              % (lab, o["RootGain_millibits"], o["PRI_bits"], o["J_source"],
                 o["n_folds"], o["n_calib_pairs"], o["n_audit_pairs"]))
        for p in sorted(o["positions"], key=int):
            e = o["positions"][p]
            ci = e["J_ci95"]
            rr = e.get("retention_ratio")
            rs = ("%.4f" % rr) if isinstance(rr, (int, float)) else str(rr)
            print("      phys%s  J=%+.6f  CI=[%+.6f,%+.6f]  R=%s"
                  % (p, e["J_point"], ci[0], ci[1], rs))
print()
print("#### comparisons (pre-registered, same seed, one direction)")
for k, h in r.get("comparisons", {}).items():
    print(" ", k)
    for hop in ("1", "2", "3"):
        if hop in h:
            e = h[hop]
            print("      hop%s (%-6s) diff=%+.6f bits (%+.3f mbit) CI=[%+.6f,%+.6f] p_sf=%.3g p_holm=%.3g sig=%s"
                  % (hop, e["line"], e["diff_bits"], e["diff_millibits"],
                     e["diff_ci95_bits"][0], e["diff_ci95_bits"][1],
                     e["p_signflip"], e.get("p_holm", float("nan")),
                     e.get("significant_05")))
print()
print("#### across-seed summary")
for k, hs in r.get("comparisons_across_seed", {}).items():
    print(" ", k)
    for hop, v in hs.items():
        print("      hop%s n_seeds=%d mean=%+.6f bits (%+.3f mbit) ci_claimed=%s"
              % (hop, v["n_seeds"], v["mean_diff_bits"], v["mean_diff_millibits"],
                 v["ci_claimed"]))
