"""Launcher for experiment 09_tuned — writes output to scratch/09_tuned_out.txt."""
import sys
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).parents[1]
OUT  = ROOT / "scratch" / "09_tuned_out.txt"
OUT.parent.mkdir(exist_ok=True)

result = subprocess.run(
    [
        str(ROOT / ".venv" / "Scripts" / "python.exe"),
        "-u", "-c",
        f"import sys; sys.path.insert(0, r'{ROOT / 'src'}'); exec(open(r'{ROOT / 'experiments' / '09_tuned.py'}').read())"
    ],
    capture_output=True,
    text=True,
    cwd=str(ROOT),
)
output = result.stdout + result.stderr
OUT.write_text(output, encoding="utf-8")
print(f"Done. Exit code: {result.returncode}")
print(f"Output: {OUT}")
