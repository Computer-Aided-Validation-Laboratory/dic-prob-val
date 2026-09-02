import sys
import subprocess
from pathlib import Path
from pyvale.mooseherder import MooseRunner


def get_file_base(moose_file: Path) -> str:
    content = moose_file.read_text()
    sim_name = None
    for line in content.splitlines():
        if "simName" in line and "=" in line:
            sim_name = line.split("=")[1].split("#")[0].strip()
            break
    if not sim_name:
        sim_name = (
            moose_file.name.replace("plate", "")
            .replace(".i", "")
        )
    end_time = "24"
    common_load_path = moose_file.parent / "common_load_time.i"
    if common_load_path.exists():
        for line in common_load_path.read_text().splitlines():
            if "endTime" in line and "=" in line:
                end_time = line.split("=")[1].split("#")[0].strip()
                break
    return f"out_{sim_name}_{end_time}f"


def run_moose_and_tee(
    moose_runner: MooseRunner, moose_path: Path
) -> None:
    file_base = get_file_base(moose_path)
    out_path = moose_path.parent / f"{file_base}.out"

    moose_runner.set_env_vars()
    moose_runner.set_stdout(False)
    arg_list = moose_runner.assemble_arg_list(moose_path)

    print(f"Running command: {' '.join(arg_list)}")
    print(f"Logging stdout to: {out_path.name}")

    with open(out_path, "w", encoding="utf-8") as out_file:
        process = subprocess.Popen(
            arg_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(moose_path.parent),
            text=True,
            bufsize=1,
        )

        try:
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    out_file.write(line)
                    out_file.flush()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise

        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode, arg_list
            )
