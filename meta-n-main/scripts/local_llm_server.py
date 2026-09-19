"""Minimal OpenAI-compatible server backed by a local transformers model.

Purpose: exercise the REAL prompt -> generation -> parser -> solver -> archive
chain with zero API cost. The model is deliberately small (1.5B); the point is
that it really generates tokens, really mis-formats them sometimes, and really
produces unstable output -- things a MockBackend cannot reproduce.

Only the endpoints Meta^n needs:

    POST /v1/chat/completions
    GET  /v1/models

`finish_reason` is reported honestly ("length" when the token budget ran out,
"stop" otherwise) because Meta^n's empty-content escalation keys off
`finish_reason == "length"` plus empty content.

Stdlib only (no FastAPI) so the environment stays small. Generation is
serialised by a lock: one GPU, one model.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = None
TOKENIZER = None
GEN_LOCK = threading.Lock()
MODEL_NAME = "local"
DEFAULTS = {"max_new_tokens": 1024, "temperature": 0.7}


def _load(path: str, load_4bit: bool = False) -> None:
    global MODEL, TOKENIZER
    print("[server] loading {} (4bit={}) ...".format(path, load_4bit),
          flush=True)
    t0 = time.time()
    TOKENIZER = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    kwargs = {"trust_remote_code": True, "device_map": "cuda"}
    if load_4bit:
        # A 30B model does not fit at bf16 in 24 GB; NF4 + bf16 compute keeps
        # it resident without a meaningful quality loss for generation.
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["dtype"] = torch.bfloat16
    MODEL = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    MODEL.eval()
    print("[server] loaded in {:.1f}s".format(time.time() - t0), flush=True)


def _generate(messages, max_tokens, temperature):
    text = TOKENIZER.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    enc = TOKENIZER([text], return_tensors="pt").to(MODEL.device)
    n_in = int(enc["input_ids"].shape[1])
    kwargs = dict(max_new_tokens=int(max_tokens or DEFAULTS["max_new_tokens"]),
                  do_sample=(temperature or 0) > 0,
                  pad_token_id=TOKENIZER.eos_token_id)
    if kwargs["do_sample"]:
        kwargs["temperature"] = float(temperature)
    with GEN_LOCK, torch.no_grad():
        out = MODEL.generate(**enc, **kwargs)
    gen = out[0][n_in:]
    n_out = int(gen.shape[0])
    content = TOKENIZER.decode(gen, skip_special_tokens=True)
    finish = "length" if n_out >= kwargs["max_new_tokens"] else "stop"
    return content, n_in, n_out, finish


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):                       # keep the console quiet
        pass

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/v1/models"):
            self._send(200, {"object": "list", "data": [
                {"id": MODEL_NAME, "object": "model", "owned_by": "local"}]})
        else:
            self._send(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(404, {"error": {"message": "not found"}})
            return
        n = int(self.headers.get("Content-Length", 0) or 0)
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError as e:
            self._send(400, {"error": {"message": "bad json: %s" % e}})
            return
        messages = req.get("messages") or []
        try:
            content, pt, ct, finish = _generate(
                messages, req.get("max_tokens"), req.get("temperature"))
        except Exception as e:                                  # noqa: BLE001
            self._send(500, {"error": {"message": "{}: {}".format(
                type(e).__name__, e)}})
            return
        self._send(200, {
            "id": "chatcmpl-local", "object": "chat.completion",
            "created": int(time.time()), "model": req.get("model", MODEL_NAME),
            "choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct,
                      "total_tokens": pt + ct},
        })


def main():
    global MODEL_NAME
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--name", default="local-model")
    ap.add_argument("--load-4bit", action="store_true",
                    help="NF4 4-bit load; needed for a 30B model on 24 GB.")
    args = ap.parse_args()
    MODEL_NAME = args.name
    _load(args.model, load_4bit=args.load_4bit)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print("[server] listening on http://{}:{}/v1".format(args.host, args.port),
          flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
