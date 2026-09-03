"""
================================================================================
simorc: simulation orchestrator
License: MIT
================================================================================
MOOSE finite element solver adapter.
"""

from dataclasses import dataclass, field
from pathlib import Path
import shutil
import subprocess
import netCDF4
import numpy as np
from .model import IModel, OutputSpec
from .param import ParamValues
from .result import EFieldLocation, ResultField, ResultScalar, SimResult


@dataclass(slots=True)
class ModelMoose(IModel):
    """MOOSE finite-element model deterministic adapter."""

    executable: Path
    input_template: Path
    mesh_template: Path | None = None
    geo_template: Path | None = None
    gmsh_path: Path | None = None
    common_files: tuple[Path, ...] = ()
    output_specs: tuple[OutputSpec, ...] = ()
    sim_name: str = "sim"
    num_threads: int = 1

    def prepare_work_dir(
        self, params: ParamValues, work_dir: Path
    ) -> tuple[Path, dict[str, float]]:
        """Prepare work directory, copy templates, substitute parameters."""
        work_dir.mkdir(parents=True, exist_ok=True)
        param_dict = params.extract_dict(sample_idx=0)

        # Copy common include files
        for c_file in self.common_files:
            if c_file.exists():
                dest_file = work_dir / c_file.name
                shutil.copy2(c_file, dest_file)

        # Handle geometry and Gmsh meshing if needed
        if self.geo_template is not None and self.geo_template.exists():
            geo_content = self.geo_template.read_text(encoding="utf-8")
            for p_name, p_val in param_dict.items():
                geo_content = self._substitute_assignment(
                    geo_content, p_name, p_val
                )
            target_geo = work_dir / self.geo_template.name
            target_geo.write_text(geo_content, encoding="utf-8")

            # Run Gmsh
            if self.gmsh_path is not None and self.gmsh_path.exists():
                subprocess.run(
                    [str(self.gmsh_path), str(target_geo)],
                    cwd=str(work_dir),
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
        elif self.mesh_template is not None and self.mesh_template.exists():
            shutil.copy2(self.mesh_template, work_dir / self.mesh_template.name)

        # Prepare main MOOSE input deck
        input_content = self.input_template.read_text(encoding="utf-8")

        # Substitute parameter assignments in main input and common files
        for p_name, p_val in param_dict.items():
            input_content = self._substitute_assignment(
                input_content, p_name, p_val
            )

        # Also substitute in copied common files in work_dir
        for c_file in self.common_files:
            target_c = work_dir / c_file.name
            if target_c.exists():
                c_text = target_c.read_text(encoding="utf-8")
                for p_name, p_val in param_dict.items():
                    c_text = self._substitute_assignment(
                        c_text, p_name, p_val
                    )
                target_c.write_text(c_text, encoding="utf-8")

        target_input = work_dir / self.input_template.name
        target_input.write_text(input_content, encoding="utf-8")

        return target_input, param_dict

    def _substitute_assignment(
        self, text: str, name: str, value: float
    ) -> str:
        """Replace standard parameter assignment 'name = value'."""
        lines = text.splitlines()
        new_lines = []
        for line in lines:
            stripped = line.strip()
            if (
                stripped.startswith(f"{name} ")
                or stripped.startswith(f"{name}=")
                or stripped.startswith(f"{name}\t")
            ):
                if "=" in stripped and not stripped.startswith("#"):
                    comment = ""
                    if "#" in line:
                        comment = "  #" + line.split("#", 1)[1]
                    elif "//" in line:
                        comment = "  //" + line.split("//", 1)[1]
                    semicolon = ";" if ";" in stripped else ""
                    new_lines.append(f"{name} = {value}{semicolon}{comment}")
                    continue
            new_lines.append(line)
        return "\n".join(new_lines) + "\n"

    def run(
        self,
        params: ParamValues,
        work_dir: Path,
    ) -> SimResult:
        """Run MOOSE simulation deterministically in work_dir."""
        target_input, _ = self.prepare_work_dir(params, work_dir)

        cmd = [
            str(self.executable),
            f"--n-threads={self.num_threads}",
            "-i",
            target_input.name,
        ]

        stdout_log = work_dir / "stdout.log"
        stderr_log = work_dir / "stderr.log"

        with open(stdout_log, "w", encoding="utf-8") as out_f, open(
            stderr_log, "w", encoding="utf-8"
        ) as err_f:
            res = subprocess.run(
                cmd,
                cwd=str(work_dir),
                stdout=out_f,
                stderr=err_f,
                check=False,
            )

        if res.returncode != 0:
            err_msg = stderr_log.read_text(encoding="utf-8")
            raise RuntimeError(
                f"MOOSE execution failed with return code {res.returncode}.\n"
                f"Log: {err_msg}"
            )

        return self.parse_outputs(work_dir)

    def parse_outputs(self, work_dir: Path) -> SimResult:
        """Parse CSV and Exodus output files from completed MOOSE run."""
        scalars_list = []
        fields_list = []

        # Find CSV file for postprocessors
        csv_files = list(work_dir.glob("*.csv"))
        if csv_files:
            csv_path = sorted(csv_files)[0]
            with open(csv_path, "r", encoding="utf-8") as f:
                header_line = f.readline().strip()
                headers = [h.strip() for h in header_line.split(",")]
                data_rows = [
                    line.strip().split(",") for line in f if line.strip()
                ]

            if data_rows:
                last_row = [float(v) for v in data_rows[-1]]
                csv_data = dict(zip(headers, last_row))
                for spec in self.output_specs:
                    if spec.output_type == "scalar":
                        if spec.target_name in csv_data:
                            scalars_list.append(
                                ResultScalar(
                                    name=spec.name,
                                    value=csv_data[spec.target_name],
                                    unit=spec.unit,
                                )
                            )
                if not self.output_specs:
                    for h_name, h_val in csv_data.items():
                        scalars_list.append(
                            ResultScalar(name=h_name, value=h_val)
                        )

        # Find Exodus file for fields
        exodus_files = list(work_dir.glob("*.e")) + list(
            work_dir.glob("*.exo")
        )
        if exodus_files:
            exo_path = sorted(exodus_files)[0]
            nc = netCDF4.Dataset(exo_path, "r")
            try:
                coord_x = np.array(nc.variables["coordx"][:], dtype=np.float64)
                coord_y = np.array(nc.variables["coordy"][:], dtype=np.float64)
                if "coordz" in nc.variables:
                    coord_z = np.array(
                        nc.variables["coordz"][:], dtype=np.float64
                    )
                    node_coords = np.column_stack(
                        [coord_x, coord_y, coord_z]
                    )
                else:
                    node_coords = np.column_stack([coord_x, coord_y])

                nod_var_names = []
                if "name_nod_var" in nc.variables:
                    raw_names = nc.variables["name_nod_var"][:]
                    for char_arr in raw_names:
                        nod_var_names.append(
                            b"".join(char_arr).decode("utf-8")
                            .split("\x00")[0].strip()
                        )

                elem_var_names = []
                if "name_elem_var" in nc.variables:
                    raw_names = nc.variables["name_elem_var"][:]
                    for char_arr in raw_names:
                        elem_var_names.append(
                            b"".join(char_arr).decode("utf-8")
                            .split("\x00")[0].strip()
                        )

                # Compute element centroids if needed
                elem_coords = None
                if "connect1" in nc.variables:
                    conn = np.array(nc.variables["connect1"][:]) - 1
                    elem_coords = np.mean(node_coords[conn], axis=1)

                for spec in self.output_specs:
                    if spec.output_type == "field":
                        if spec.target_name in nod_var_names:
                            var_idx = (
                                nod_var_names.index(spec.target_name) + 1
                            )
                            vals = nc.variables[f"vals_nod_var{var_idx}"][-1]
                            fields_list.append(
                                ResultField(
                                    name=spec.name,
                                    values=np.array(vals, dtype=np.float64),
                                    coords=node_coords,
                                    components=(spec.name,),
                                    location=EFieldLocation.node,
                                )
                            )
                        elif spec.target_name in elem_var_names:
                            var_idx = (
                                elem_var_names.index(spec.target_name) + 1
                            )
                            vals = nc.variables[
                                f"vals_elem_var{var_idx}eb1"
                            ][-1]
                            coords_to_use = (
                                elem_coords
                                if elem_coords is not None
                                else node_coords
                            )
                            fields_list.append(
                                ResultField(
                                    name=spec.name,
                                    values=np.array(vals, dtype=np.float64),
                                    coords=coords_to_use,
                                    components=(spec.name,),
                                    location=EFieldLocation.element,
                                )
                            )
            finally:
                nc.close()

        return SimResult(scalars=scalars_list, fields=fields_list)
