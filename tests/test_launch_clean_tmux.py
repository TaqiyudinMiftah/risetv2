from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCH_SCRIPT = REPO_ROOT / "scripts" / "launch_clean_tmux.sh"


class CleanTmuxLauncherTests(unittest.TestCase):
    def _capture_launch_command(self, extra_environment: dict[str, str] | None = None) -> str:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            binary_directory = root / "bin"
            binary_directory.mkdir()
            capture_path = root / "tmux-command.txt"
            tmux_path = binary_directory / "tmux"
            tmux_path.write_text(
                "#!/usr/bin/env bash\n"
                "if [[ \"$1\" == \"has-session\" ]]; then exit 1; fi\n"
                "if [[ \"$1\" == \"new-session\" ]]; then\n"
                "  printf '%s\\n' \"$@\" > \"$TMUX_CAPTURE\"\n"
                "  exit 0\n"
                "fi\n"
                "exit 2\n",
                encoding="utf-8",
            )
            tmux_path.chmod(0o755)
            environment = dict(os.environ)
            environment.update(
                {
                    "PATH": f"{binary_directory}:{environment['PATH']}",
                    "TMUX_CAPTURE": str(capture_path),
                    "TMUX_SESSION": "clean-launch-fixture",
                    "CONFIG": "configs/experiments/caernet_clean_input_ablation_face_only_content_disjoint_exploratory_seed42.json",
                    "DEVICE_IDS": "0",
                    "N_GPU": "1",
                    "SEED": "42",
                }
            )
            for key in ("RUN_ID", "HSA_OVERRIDE_GFX_VERSION", "ROCR_VISIBLE_DEVICES"):
                environment.pop(key, None)
            if extra_environment is not None:
                environment.update(extra_environment)
            subprocess.run([str(LAUNCH_SCRIPT)], cwd=REPO_ROOT, env=environment, check=True)
            arguments = capture_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(arguments[:4], ["new-session", "-d", "-s", "clean-launch-fixture"])
        return arguments[4]

    def test_script_uses_variant_aware_generated_run_id_and_rocm_defaults(self) -> None:
        command = self._capture_launch_command()

        self.assertIn("HSA_OVERRIDE_GFX_VERSION=10.3.0", command)
        self.assertIn("ROCR_VISIBLE_DEVICES=0", command)
        self.assertIn("--config", command)
        self.assertNotIn("--run-id", command)
        self.assertNotIn("caernet__clean_inrepo", command)

    def test_script_forwards_only_an_explicit_run_id(self) -> None:
        run_id = "caernet__input_ablation_face_only_exploratory__seed42__fixture"
        command = self._capture_launch_command({"RUN_ID": run_id})

        self.assertIn(f"--run-id {run_id}", command)
