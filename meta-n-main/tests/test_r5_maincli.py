"""R5 main.py CLI seams: deploy-verified-code language-gate note (D-handoff)
and the F075 live-site adapter wiring (C-handoff).
"""

from meta_n.main import build_parser, deploy_verified_code_noop_note


def _parse(argv):
    return build_parser().parse_args(argv)


class TestDeployVerifiedCodeNoopNote:
    def test_bash_language_is_noted(self):
        ns = _parse(["--deploy-verified-code", "--use-archive", "--verified-code"])
        note = deploy_verified_code_noop_note(ns, "bash")
        assert note is not None
        assert note.startswith("--deploy-verified-code")
        assert "bash" in note

    def test_openevolve_language_is_noted(self):
        ns = _parse(["--deploy-verified-code", "--use-archive", "--verified-code"])
        note = deploy_verified_code_noop_note(ns, "openevolve")
        assert note is not None and "openevolve" in note

    def test_python_language_not_noted(self):
        ns = _parse(["--deploy-verified-code", "--use-archive", "--verified-code"])
        assert deploy_verified_code_noop_note(ns, "python") is None

    def test_flag_off_not_noted(self):
        ns = _parse(["--use-archive"])
        assert deploy_verified_code_noop_note(ns, "bash") is None


class TestTBBaseSolverDeclaredAtConstruction:
    def test_both_tb_adapters_receive_base_solver(self):
        # Pre-spend legacy-layout refusal needs the declared route at
        # construction time (load_tasks raises before any solver call).
        import inspect

        import meta_n.main as main_mod

        src = inspect.getsource(main_mod)
        for marker in (
            "adapter = TerminalBenchAdapter(",
            "adapter = SWEBenchVerifiedAdapter(",
        ):
            window = src.split(marker)[1][:400]
            assert "base_solver=args.base_solver" in window, marker


class TestClassifyFallbackLiveWiring:
    def test_adapter_ctor_receives_the_flag(self):
        # The live parse site is the adapter (F075); main.py must thread the
        # CLI flag into the TextClassificationAdapter construction.
        import inspect

        import meta_n.main as main_mod

        src = inspect.getsource(main_mod)
        adapter_call = src.split("adapter = TextClassificationAdapter(")[1][:800]
        assert (
            "balanced_json_fallback=args.classify_balanced_json_fallback"
            in adapter_call
        )
