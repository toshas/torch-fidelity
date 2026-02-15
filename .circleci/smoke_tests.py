import json
import os
import psutil
import subprocess
import time
import unittest
from pathlib import Path

from tests import TimeTrackingTestCase


def get_memory_usage():
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss / (1024 * 1024)
    except Exception:
        return None


def get_system_memory():
    try:
        mem = psutil.virtual_memory()
        return {
            "total_mb": mem.total / (1024 * 1024),
            "available_mb": mem.available / (1024 * 1024),
            "used_mb": mem.used / (1024 * 1024),
            "percent": mem.percent,
        }
    except Exception:
        return None


def log_resource_info(label):
    mem_usage = get_memory_usage()
    sys_mem = get_system_memory()
    print(f"\n=== {label} ===")
    if mem_usage is not None:
        print(f"Process memory: {mem_usage:.2f} MB")
    if sys_mem is not None:
        print(
            f"System memory: {sys_mem['used_mb']:.2f} MB / {sys_mem['total_mb']:.2f} MB "
            f"({sys_mem['percent']:.1f}% used, {sys_mem['available_mb']:.2f} MB available)"
        )
    print("=" * 50)


class SmokeTests(TimeTrackingTestCase):
    def setUp(self):
        super().setUp()
        log_resource_info("Test setup - Initial state")

    def _run_fidelity_command(self, cmd_args, test_name, timeout=None):
        """Run fidelity command with diagnostics and timeout."""
        log_resource_info(f"Before {test_name}")
        
        start_time = time.time()
        try:
            res = subprocess.run(
                cmd_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
            elapsed = time.time() - start_time
        except subprocess.TimeoutExpired as e:
            elapsed = time.time() - start_time
            log_resource_info(f"During {test_name} (TIMEOUT after {elapsed:.1f}s)")
            print(f"\nTIMEOUT: Command exceeded {timeout}s timeout")
            print(f"STDOUT (partial):\n{e.stdout}\n")
            print(f"STDERR (partial):\n{e.stderr}\n")
            raise
        except Exception as e:
            elapsed = time.time() - start_time
            log_resource_info(f"During {test_name} (EXCEPTION after {elapsed:.1f}s)")
            print(f"\nEXCEPTION during subprocess execution: {type(e).__name__}: {e}")
            raise

        log_resource_info(f"After {test_name} (completed in {elapsed:.1f}s)")
        
        if res.returncode == -9:
            print("\n=== Last 1000 lines of dmesg ===")
            try:
                dmesg_res = subprocess.run(
                    ["dmesg"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=10,
                )
                if dmesg_res.returncode == 0:
                    lines = dmesg_res.stdout.splitlines()
                    print(f"Total dmesg lines: {len(lines)}")
                    print("Last 1000 lines:")
                    print("-" * 80)
                    for line in lines[-1000:]:
                        print(line)
                    print("-" * 80)
                else:
                    print(f"dmesg failed with return code {dmesg_res.returncode}")
                    print(f"dmesg stderr: {dmesg_res.stderr}")
            except subprocess.TimeoutExpired:
                print("dmesg command timed out after 10 seconds")
            except Exception as e:
                print(f"Failed to run dmesg: {type(e).__name__}: {e}")
            
            # Also try kern.log if accessible
            print("\n=== Last 1000 lines of /var/log/kern.log (if accessible) ===")
            try:
                with open("/var/log/kern.log", "r") as f:
                    lines = f.readlines()
                    print(f"Total kern.log lines: {len(lines)}")
                    print("Last 1000 lines:")
                    print("-" * 80)
                    for line in lines[-1000:]:
                        print(line.rstrip())
                    print("-" * 80)
            except FileNotFoundError:
                print("/var/log/kern.log not found (normal in containers)")
            except PermissionError:
                print("/var/log/kern.log permission denied")
            except Exception as e:
                print(f"Failed to read kern.log: {type(e).__name__}: {e}")
        
        print(f"\nRETCODE: {res.returncode}")
        print(f"STDOUT:\n{res.stdout}")
        print(f"STDERR:\n{res.stderr}")
        print()
        
        return res

    def test_latest(self):
        path_base = Path(__file__).parent.parent.parent / "data"
        path_input1 = str(path_base / "cifar10-train-256")
        path_input2 = str(path_base / "cifar10-valid-256")

        res = self._run_fidelity_command(
            (
                # fmt: off
                "python3", "-m", "torch_fidelity.fidelity",
                "--input1", path_input1,
                "--input2", path_input2,
                "--isc",
                "--fid",
                "--kid",
                "--prc",
                "--kid-subset-size", "64",
                "--silent",
                "--json",
                # fmt: on
            ),
            test_name="test_inception_all_metrics",
            timeout=1200,
        )
        self.assertEqual(res.returncode, 0, msg="Non-zero return code")
        self.assertTrue("Warning" not in res.stdout, msg="Warning in stdout")
        self.assertTrue("Warning" not in res.stderr, msg="Warning in stderr")
        metrics = json.loads(res.stdout)
        self.assertAlmostEqual(metrics["inception_score_mean"], 6.675409920681458, delta=1e-3)
        self.assertAlmostEqual(metrics["inception_score_std"], 0.9399683668381174, delta=1e-3)
        self.assertAlmostEqual(metrics["frechet_inception_distance"], 110.28082617202443, delta=1e-2)
        self.assertAlmostEqual(metrics["kernel_inception_distance_mean"], -0.0006792521855187905, delta=1e-4)
        self.assertAlmostEqual(metrics["kernel_inception_distance_std"], 0.0017778231588294379, delta=1e-4)
        self.assertAlmostEqual(metrics["precision"], 0.71484375, delta=1e-3)
        self.assertAlmostEqual(metrics["recall"], 0.7109375, delta=1e-3)
        self.assertAlmostEqual(metrics["f_score"], 0.7128852739726027, delta=1e-3)

        res = self._run_fidelity_command(
            (
                # fmt: off
                "python3", "-m", "torch_fidelity.fidelity",
                "--input1", path_input1,
                "--isc",
                "--silent",
                "--json",
                "--feature-extractor", "clip-vit-b-32",
                # fmt: on
            ),
            test_name="test_clip_vit_b_32",
            timeout=1200,
        )
        self.assertEqual(res.returncode, 0, msg="Non-zero return code")
        self.assertTrue("Warning" not in res.stdout, msg="Warning in stdout")
        self.assertTrue("Warning" not in res.stderr, msg="Warning in stderr")
        metrics = json.loads(res.stdout)
        self.assertAlmostEqual(metrics["inception_score_mean"], 1.0322264848429967, delta=1e-3)
        self.assertAlmostEqual(metrics["inception_score_std"], 0.0012455888960011387, delta=1e-3)

        res = self._run_fidelity_command(
            (
                # fmt: off
                "python3", "-m", "torch_fidelity.fidelity",
                "--input1", path_input1,
                "--isc",
                "--silent",
                "--json",
                "--feature-extractor", "dinov2-vit-b-14",
                # fmt: on
            ),
            test_name="test_dinov2_vit_b_14",
            timeout=1200,
        )
        self.assertEqual(res.returncode, 0, msg="Non-zero return code")
        self.assertTrue("Warning" not in res.stdout, msg="Warning in stdout")
        self.assertTrue("Warning" not in res.stderr, msg="Warning in stderr")
        metrics = json.loads(res.stdout)
        self.assertAlmostEqual(metrics["inception_score_mean"], 3.2955061842255757, delta=1e-3)
        self.assertAlmostEqual(metrics["inception_score_std"], 0.23961402932647136, delta=1e-3)



if __name__ == "__main__":
    unittest.main()
