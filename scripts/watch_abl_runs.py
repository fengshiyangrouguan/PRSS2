"""本机侧消融结果监听器（配合 Monitor 工具使用）。

每 POLL_SEC 秒轮询三卡一次，**只在状态发生变化时**向 stdout 打印一行事件；
Monitor 把每一行 stdout 转成一条通知，所以输出必须少而准：

  ✅ [A1] 完成 P0 s0 | bestAP=0.9178 testAP=... AUC=... ep=76
  ❌ [A2A6] 失败 S2 s0 rc=1
  🏁 [A3A7] 全部完成 (12/12)
  ⚠️ [A1] 异常：训练进程消失且该组未完成（连续 2 次确认）
  🔌 [A2A6] 链路异常：SSH 不可达（连续 2 次确认）

首轮静默建立基线，不把既有的 summary.json 全报一遍。
"""
import json
import os
import subprocess
import sys
import time

# Windows 中文环境下 stdout 默认 GBK，emoji 会炸；强制 UTF-8（Monitor 按字节读）
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

POLL = int(os.environ.get("POLL_SEC", "300"))
HOST = "connect.weste.seetacloud.com"
GROUPS = [("A1", 35360), ("A2A6", 17127), ("A3A7", 26389)]
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
       "-o", "StrictHostKeyChecking=no", "-o", "LogLevel=ERROR"]


def emit(line):
    print(line, flush=True)


def fetch(grp, port):
    """返回远端状态块文本；失败返回 None。"""
    try:
        r = subprocess.run(
            SSH + ["-p", str(port), "root@" + HOST, "bash /root/watch_abl.sh " + grp],
            capture_output=True, text=True, timeout=60)
        return r.stdout if r.returncode == 0 else None
    except Exception:
        return None


def parse(txt):
    runs, dones, info = {}, [], {}
    for line in txt.splitlines():
        if line.startswith("RUN|"):
            p = line.split("|")
            runs["%s/%s" % (p[1], p[2])] = p[3:]
        elif "=" in line and not line.startswith("---"):
            k, _, v = line.partition("=")
            if k in ("PROCS", "ORCH", "ALLDONE", "NOW"):
                info[k] = v
        elif any(t in line for t in ("DONE ", "ALL_DONE", "BUSY ", "RESET ")):
            dones.append(line.strip())
    return runs, dones, info


def label(key):
    parts = key.split("/")
    return "%s s%s" % (parts[1], parts[0].replace("seed", ""))


def main():
    state = {}
    first = True
    emit("👀 消融结果监听已启动（轮询 %d 秒，静默建立基线中…）" % POLL)
    while True:
        for grp, port in GROUPS:
            txt = fetch(grp, port)
            if txt is None:
                st = state.setdefault(grp, {})
                st["ssh_fail"] = st.get("ssh_fail", 0) + 1
                if st["ssh_fail"] == 2 and not first:
                    emit("🔌 [%s] 链路异常：SSH 不可达（连续 2 次）" % grp)
                continue

            runs, dones, info = parse(txt)
            st = state.setdefault(grp, {})
            st["ssh_fail"] = 0
            prev_runs = st.get("runs")
            prev_dones = st.get("dones")
            done_all = info.get("ALLDONE", "0") not in ("0", "")
            procs = int(info.get("PROCS", "-1") or -1)
            orch = int(info.get("ORCH", "-1") or -1)

            if first:
                st["runs"], st["dones"] = set(runs), set(dones)
                st["anom"] = 0
                st["alldone"] = done_all
                continue

            # 1) 新增完成的 run
            for k in sorted(set(runs) - (prev_runs or set())):
                v = runs[k]
                if v and v[0] == "PARSE_ERR":
                    emit("❓ [%s] %s 产出 summary.json 但解析失败" % (grp, label(k)))
                else:
                    emit("✅ [%s] 完成 %s | bestAP=%s testAP=%s AUC=%s ep=%s %s"
                         % (grp, label(k), v[0], v[1], v[2], v[3],
                            ("[%s]" % v[5]) if len(v) > 5 and v[5] not in ("-", "None") else ""))

            # 2) 新增的失败 / 异常日志行
            for d in [x for x in dones if x not in (prev_dones or set())]:
                if "DONE " in d and "rc=0" not in d:
                    emit("❌ [%s] %s" % (grp, d))

            # 3) 全部完成
            if done_all and not st.get("alldone"):
                emit("🏁 [%s] 该组全部完成（%d 个 summary 在册）" % (grp, len(runs)))
                st["alldone"] = True

            # 4) 进程异常（连续 2 次确认，避开 run 切换的几秒空隙）
            bad = None
            if procs == 0 and orch > 0 and not done_all:
                bad = "训练进程消失但编排器仍在、且该组未完成"
            elif orch == 0 and not done_all:
                bad = "编排器进程消失且该组未完成（需人工介入）"
            if bad:
                st["anom"] = st.get("anom", 0) + 1
                if st["anom"] == 2:
                    emit("⚠️ [%s] %s | 已完成 %d 个" % (grp, bad, len(runs)))
            else:
                st["anom"] = 0

            st["runs"], st["dones"] = set(runs), set(dones)

        first = False
        time.sleep(POLL)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
