import os
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.io import loadmat


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_FOLDER = Path(__file__).resolve().parent

# MATLAB point-cloud files
POINTS_FILE = PROJECT_FOLDER / "points.mat"
FACES_FILE = PROJECT_FOLDER / "facesShapeModel.mat"

LEFT_ARM_INDEX_FILE = PROJECT_FOLDER / "idxLA.mat"
RIGHT_ARM_INDEX_FILE = PROJECT_FOLDER / "idxRA.mat"
LEFT_LEG_INDEX_FILE = PROJECT_FOLDER / "idxLL.mat"
RIGHT_LEG_INDEX_FILE = PROJECT_FOLDER / "idxRL.mat"
HEAD_INDEX_FILE = PROJECT_FOLDER / "idxHead.mat"
TRUNK_INDEX_FILE = PROJECT_FOLDER / "TotalIdx.mat"

# OpenSim XML files
TEMPLATE_XML = PROJECT_FOLDER / "ScaleSetup_PointCloud_Template.xml"
OUTPUT_XML = PROJECT_FOLDER / "P2_Scaled_ScaleSetup.xml"
OUTPUT_TXT = PROJECT_FOLDER / "subject_measurements.txt"

# OpenSim model files
ORIGINAL_MODEL = PROJECT_FOLDER / "P2_APP1_SO.osim"
SCALED_MODEL = PROJECT_FOLDER / "P2_PointCloud_scaled.osim"
GEOMETRY_FOLDER = PROJECT_FOLDER / "Geometry"

# Uniform tissue density:
# 1000 kg/m^3 = 1e-6 kg/mm^3
DENSITY_KG_PER_MM3 = 1e-6

# Translation used to verify that volume is independent of mesh position.
VOLUME_TEST_SHIFT_MM = np.array([500.0, -300.0, 800.0])


# ============================================================
# LOAD A VARIABLE FROM A MATLAB FILE
# ============================================================

def load_mat_variable(file_path, variable_name):
    """Load one variable from a standard MATLAB .mat file."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required MATLAB file was not found:\n{file_path}"
        )

    try:
        mat_data = loadmat(file_path)
    except NotImplementedError as error:
        raise RuntimeError(
            f"{file_path.name} appears to use MATLAB v7.3 format.\n"
            "SciPy cannot read that format using loadmat().\n"
            "Resave it in MATLAB using a standard MAT format, or use h5py/mat73."
        ) from error

    if variable_name not in mat_data:
        available = [key for key in mat_data if not key.startswith("__")]
        raise KeyError(
            f'The variable "{variable_name}" was not found in '
            f"{file_path.name}.\nAvailable variables: {available}"
        )

    return np.asarray(mat_data[variable_name])


# ============================================================
# VALIDATION AND INDEX CONVERSION
# ============================================================

def validate_inputs(points, faces):
    """Validate the complete point and face matrices."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be an N x 3 XYZ matrix.")

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must be an M x 3 triangle-index matrix.")

    if not np.all(np.isfinite(points)):
        raise ValueError("The point cloud contains NaN or infinite values.")

    if not np.all(np.isfinite(faces)):
        raise ValueError("The face matrix contains NaN or infinite values.")

    if not np.allclose(faces, np.round(faces)):
        raise ValueError("The face matrix contains non-integer indices.")

    if np.any(faces < 1) or np.any(faces > len(points)):
        raise IndexError(
            f"faces contains values outside MATLAB's valid point range "
            f"1 to {len(points)}."
        )


def convert_matlab_indices(index_array, maximum_size, index_name):
    """Convert one-based MATLAB indices into zero-based Python indices."""
    indices = np.asarray(index_array).reshape(-1)

    if indices.size == 0:
        raise ValueError(f"{index_name} contains no indices.")

    if not np.all(np.isfinite(indices)):
        raise ValueError(f"{index_name} contains invalid values.")

    if not np.allclose(indices, np.round(indices)):
        raise ValueError(f"{index_name} contains non-integer values.")

    indices = indices.astype(np.int64)

    if np.any(indices < 1) or np.any(indices > maximum_size):
        raise IndexError(
            f"{index_name} contains values outside MATLAB's valid range "
            f"1 to {maximum_size}."
        )

    return indices - 1


# ============================================================
# SEGMENT MESH EXTRACTION AND CAPPING
# ============================================================

def extract_local_segment_mesh(points, faces_python, segment_face_indices):
    """Extract selected faces and remap their vertices to local indices."""
    selected_faces_global = faces_python[segment_face_indices]
    used_global_vertices = np.unique(selected_faces_global.reshape(-1))

    segment_points = points[used_global_vertices].astype(np.float64).copy()

    global_to_local = np.full(len(points), -1, dtype=np.int64)
    global_to_local[used_global_vertices] = np.arange(len(used_global_vertices))
    segment_faces = global_to_local[selected_faces_global]

    if np.any(segment_faces < 0):
        raise RuntimeError("Failed to remap the segment mesh to local indices.")

    return segment_points, segment_faces.astype(np.int64)


def boundary_edges_from_faces(faces):
    """Return directed edges that belong to only one triangle."""
    directed = np.vstack(
        (
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        )
    )
    undirected = np.sort(directed, axis=1)
    _, inverse, counts = np.unique(
        undirected,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )

    if np.any(counts > 2):
        raise RuntimeError(
            "The segment contains non-manifold edges shared by more than two faces."
        )

    return directed[counts[inverse] == 1]


def order_boundary_loops(boundary_edges):
    """Arrange directed boundary edges into closed vertex loops."""
    remaining = [tuple(map(int, edge)) for edge in boundary_edges]
    loops = []

    while remaining:
        start, current = remaining.pop(0)
        loop = [start]

        while current != start:
            loop.append(current)

            direct_match = next(
                (i for i, edge in enumerate(remaining) if edge[0] == current),
                None,
            )

            if direct_match is not None:
                _, next_vertex = remaining.pop(direct_match)
                current = next_vertex
                continue

            # Fall back to reversing an edge if the source mesh has locally
            # inconsistent winding. The final watertightness and volume tests
            # still determine whether the capped result is usable.
            reverse_match = next(
                (i for i, edge in enumerate(remaining) if edge[1] == current),
                None,
            )

            if reverse_match is None:
                raise RuntimeError(
                    "Could not form a closed boundary loop. The selected segment "
                    "may contain a broken or branching boundary."
                )

            next_vertex, _ = remaining.pop(reverse_match)
            current = next_vertex

            if len(loop) > len(boundary_edges) + 1:
                raise RuntimeError("Boundary-loop traversal did not terminate.")

        if len(loop) < 3:
            raise RuntimeError("A boundary opening has fewer than three vertices.")

        loops.append(loop)

    return loops


def cap_segment_mesh(segment_points, segment_faces):
    """Close every open segment boundary using a centroid triangle fan."""
    boundary_edges = boundary_edges_from_faces(segment_faces)

    if len(boundary_edges) == 0:
        return segment_points, segment_faces, 0

    loops = order_boundary_loops(boundary_edges)
    closed_points = segment_points.copy()
    closed_faces = segment_faces.copy()

    for loop in loops:
        loop_array = np.asarray(loop, dtype=np.int64)
        cap_center = np.mean(closed_points[loop_array], axis=0)
        center_index = len(closed_points)
        closed_points = np.vstack((closed_points, cap_center))

        next_vertices = np.roll(loop_array, -1)

        # Reverse each boundary edge for the cap, matching the MATLAB code:
        # [b, a, center].
        cap_faces = np.column_stack(
            (
                next_vertices,
                loop_array,
                np.full(len(loop_array), center_index, dtype=np.int64),
            )
        )
        closed_faces = np.vstack((closed_faces, cap_faces))

    remaining_boundaries = boundary_edges_from_faces(closed_faces)
    if len(remaining_boundaries) != 0:
        raise RuntimeError(
            f"Mesh capping failed; {len(remaining_boundaries)} boundary edges remain."
        )

    return closed_points, closed_faces.astype(np.int64), len(loops)


# ============================================================
# DIVERGENCE-THEOREM MASS PROPERTIES
# ============================================================

def calculate_mass_properties(points, faces, density_kg_per_mm3):
    """
    Calculate volume, mass, center of mass, and inertia for a closed
    triangular mesh using the divergence theorem/signed tetrahedra.
    """
    if len(faces) == 0:
        raise ValueError("The closed segment mesh contains no faces.")

    v0 = points[faces[:, 0]]
    v1 = points[faces[:, 1]]
    v2 = points[faces[:, 2]]

    signed_terms = np.einsum("ij,ij->i", v0, np.cross(v1, v2))
    signed_sum = float(np.sum(signed_terms))

    if signed_sum < 0:
        v1, v2 = v2.copy(), v1.copy()
        signed_terms = -signed_terms

    volume_mm3 = float(np.sum(signed_terms) / 6.0)

    if not np.isfinite(volume_mm3) or volume_mm3 <= 0:
        raise ValueError(
            "Divergence-theorem volume is not positive. Check mesh closure "
            "and triangle orientation."
        )

    center_of_mass_mm = np.sum(
        signed_terms[:, None] * (v0 + v1 + v2),
        axis=0,
    ) / (24.0 * volume_mm3)

    canonical = np.array(
        [[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]],
        dtype=np.float64,
    ) / 120.0

    second_moment = np.zeros((3, 3), dtype=np.float64)

    for index, signed_term in enumerate(signed_terms):
        triangle_matrix = np.column_stack((v0[index], v1[index], v2[index]))
        second_moment += (
            signed_term
            * triangle_matrix
            @ canonical
            @ triangle_matrix.T
        )

    second_moment_about_com = (
        second_moment
        - volume_mm3 * np.outer(center_of_mass_mm, center_of_mass_mm)
    ) * density_kg_per_mm3

    inertia_tensor_kg_m2 = (
        np.trace(second_moment_about_com) * np.eye(3)
        - second_moment_about_com
    ) / 1_000_000.0

    mass_kg = volume_mm3 * density_kg_per_mm3

    return {
        "volume_mm3": volume_mm3,
        "mass_kg": float(mass_kg),
        "center_of_mass_mm": center_of_mass_mm,
        "inertia_tensor_kg_m2": inertia_tensor_kg_m2,
    }


def translation_invariance_test(points, faces, original_properties):
    """Confirm that translating the mesh does not materially change volume."""
    shifted = calculate_mass_properties(
        points + VOLUME_TEST_SHIFT_MM,
        faces,
        DENSITY_KG_PER_MM3,
    )

    drift_mm3 = abs(
        original_properties["volume_mm3"] - shifted["volume_mm3"]
    )
    tolerance_mm3 = max(
        1e-6,
        original_properties["volume_mm3"] * 1e-10,
    )

    if drift_mm3 > tolerance_mm3:
        raise RuntimeError(
            "The divergence-theorem translation test failed. "
            f"Volume drift = {drift_mm3:.6e} mm^3; "
            f"allowed = {tolerance_mm3:.6e} mm^3."
        )

    return float(drift_mm3)


# ============================================================
# CALCULATE ONE BODY SEGMENT
# ============================================================

def calculate_segment(segment_name, points, faces_python, matlab_face_indices):
    """Calculate one capped segment using the divergence theorem."""
    segment_face_indices = convert_matlab_indices(
        matlab_face_indices,
        len(faces_python),
        f"{segment_name} face indices",
    )

    segment_points, segment_faces = extract_local_segment_mesh(
        points,
        faces_python,
        segment_face_indices,
    )

    if len(segment_points) < 4:
        raise ValueError(
            f"{segment_name} has fewer than four points; enclosed volume "
            "cannot be calculated."
        )

    # Preserve the current length definition until body scaling is added.
    raw_segment_length_mm = float(
        np.max(segment_points[:, 2]) - np.min(segment_points[:, 2])
    )

    closed_points, closed_faces, capped_holes = cap_segment_mesh(
        segment_points,
        segment_faces,
    )

    properties = calculate_mass_properties(
        closed_points,
        closed_faces,
        DENSITY_KG_PER_MM3,
    )

    drift_mm3 = translation_invariance_test(
        closed_points,
        closed_faces,
        properties,
    )

    if raw_segment_length_mm <= 0:
        raise ValueError(f"{segment_name} produced an invalid length.")

    return {
        "name": segment_name,
        "point_count": int(len(segment_points)),
        "original_face_count": int(len(segment_faces)),
        "closed_face_count": int(len(closed_faces)),
        "capped_holes": int(capped_holes),
        "length_mm": raw_segment_length_mm,
        "volume_mm3": properties["volume_mm3"],
        "mass_kg": properties["mass_kg"],
        "center_of_mass_mm": properties["center_of_mass_mm"],
        "inertia_tensor_kg_m2": properties["inertia_tensor_kg_m2"],
        "volume_drift_mm3": drift_mm3,
    }


# ============================================================
# LOAD ALL DATA AND CALCULATE BODY MEASUREMENTS
# ============================================================

def calculate_body_measurements():
    """Load point-cloud data and calculate all body-segment properties."""
    print("=" * 75)
    print("FULLY AUTOMATED PYTHON-OPENSIM PIPELINE")
    print("VOLUME METHOD: DIVERGENCE THEOREM")
    print("=" * 75)

    print("\n[1/7] Loading point-cloud data...")

    points = load_mat_variable(POINTS_FILE, "points").astype(np.float64)
    faces_matlab = load_mat_variable(FACES_FILE, "faces").astype(np.int64)

    idx_la = load_mat_variable(LEFT_ARM_INDEX_FILE, "idxLA")
    idx_ra = load_mat_variable(RIGHT_ARM_INDEX_FILE, "idxRA")
    idx_ll = load_mat_variable(LEFT_LEG_INDEX_FILE, "idxLL")
    idx_rl = load_mat_variable(RIGHT_LEG_INDEX_FILE, "idxRL")
    idx_head = load_mat_variable(HEAD_INDEX_FILE, "idxHead")
    total_idx = load_mat_variable(TRUNK_INDEX_FILE, "TotalIdx")

    validate_inputs(points, faces_matlab)
    faces_python = faces_matlab - 1

    print(f"Loaded {len(points):,} point-cloud vertices.")
    print(f"Loaded {len(faces_python):,} triangular faces.")

    print("\n[2/7] Capping segments and calculating mass properties...")

    segment_inputs = {
        "LEFT_ARM": ("Left Arm", idx_la),
        "RIGHT_ARM": ("Right Arm", idx_ra),
        "LEFT_LEG": ("Left Leg", idx_ll),
        "RIGHT_LEG": ("Right Leg", idx_rl),
        "HEAD": ("Head", idx_head),
        "TRUNK": ("Trunk", total_idx),
    }

    segments = {
        key: calculate_segment(name, points, faces_python, indices)
        for key, (name, indices) in segment_inputs.items()
    }

    total_mass = float(sum(item["mass_kg"] for item in segments.values()))
    total_volume = float(
        sum(item["volume_mm3"] for item in segments.values())
    )

    if not np.isfinite(total_mass) or total_mass <= 0:
        raise ValueError("Calculated total mass is invalid.")

    print("\nSegment results:")
    for segment in segments.values():
        com = segment["center_of_mass_mm"]
        print(
            f"  {segment['name']:<10} "
            f"{segment['mass_kg']:>8.2f} kg | "
            f"Length: {segment['length_mm']:>8.1f} mm | "
            f"Volume: {segment['volume_mm3']:>13.2f} mm^3 | "
            f"Caps: {segment['capped_holes']:>2} | "
            f"Drift: {segment['volume_drift_mm3']:.2e} mm^3 | "
            f"COM: [{com[0]:.1f}, {com[1]:.1f}, {com[2]:.1f}] mm"
        )

    print("-" * 75)
    print(f"TOTAL VOLUME: {total_volume:.2f} mm^3")
    print(f"TOTAL MASS:   {total_mass:.2f} kg")

    return segments, total_volume, total_mass


# ============================================================
# WRITE MEASUREMENT REPORT
# ============================================================

def write_measurement_report(segments, total_volume, total_mass):
    """Write calculated segment properties to a plain-text report."""
    print("\n[3/7] Writing measurement report...")

    with OUTPUT_TXT.open("w", encoding="utf-8") as file:
        file.write("STATUS=SUCCESS\n")
        file.write("SOURCE=PYTHON_POINT_CLOUD_PIPELINE\n")
        file.write("VOLUME_METHOD=DIVERGENCE_THEOREM_CLOSED_TRIANGLE_MESH\n")
        file.write("MASS_UNIT=kg\n")
        file.write("LENGTH_UNIT=mm\n")
        file.write("VOLUME_UNIT=mm3\n")
        file.write("COM_UNIT=mm\n")
        file.write("INERTIA_UNIT=kg_m2\n")
        file.write("LENGTH_METHOD=Z_AXIS_RANGE\n")
        file.write(f"DENSITY_KG_PER_MM3={DENSITY_KG_PER_MM3:.10f}\n")
        file.write(f"TOTAL_MASS={total_mass:.6f}\n")
        file.write(f"TOTAL_VOLUME={total_volume:.6f}\n")

        for key, segment in segments.items():
            com = segment["center_of_mass_mm"]
            inertia = segment["inertia_tensor_kg_m2"]

            file.write(f"\n[{key}]\n")
            file.write(f"NAME={segment['name']}\n")
            file.write(f"MASS={segment['mass_kg']:.10f}\n")
            file.write(f"LENGTH={segment['length_mm']:.10f}\n")
            file.write(f"VOLUME={segment['volume_mm3']:.10f}\n")
            file.write(f"POINT_COUNT={segment['point_count']}\n")
            file.write(f"ORIGINAL_FACE_COUNT={segment['original_face_count']}\n")
            file.write(f"CLOSED_FACE_COUNT={segment['closed_face_count']}\n")
            file.write(f"CAPPED_HOLES={segment['capped_holes']}\n")
            file.write(f"VOLUME_DRIFT_MM3={segment['volume_drift_mm3']:.12e}\n")
            file.write(
                "COM_MM="
                + ",".join(f"{value:.10f}" for value in com)
                + "\n"
            )
            file.write(
                "INERTIA_TENSOR_KG_M2="
                + ",".join(f"{value:.12e}" for value in inertia.reshape(-1))
                + "\n"
            )

    print(f"Generated: {OUTPUT_TXT.name}")


# ============================================================
# UPDATE AND VERIFY OPENSIM XML
# ============================================================

def find_first_xml_element(root, local_tag_name):
    """Find an XML element while tolerating optional namespaces."""
    for element in root.iter():
        if element.tag.split("}")[-1] == local_tag_name:
            return element
    return None


def generate_opensim_xml(total_mass):
    """Insert the calculated body mass in the OpenSim ScaleTool template."""
    print("\n[4/7] Generating OpenSim XML...")

    if not TEMPLATE_XML.exists():
        raise FileNotFoundError(
            f"OpenSim template was not found:\n{TEMPLATE_XML}"
        )

    tree = ET.parse(TEMPLATE_XML)
    root = tree.getroot()
    mass_element = find_first_xml_element(root, "mass")

    if mass_element is None:
        raise RuntimeError("No <mass> element was found in the OpenSim XML.")

    previous_mass = mass_element.text
    mass_element.text = f"{total_mass:.6f}"

    if hasattr(ET, "indent"):
        ET.indent(tree, space="\t", level=0)

    tree.write(OUTPUT_XML, encoding="UTF-8", xml_declaration=True)

    print(f"Old XML mass: {previous_mass} kg")
    print(f"New XML mass: {total_mass:.6f} kg")
    print(f"Generated: {OUTPUT_XML.name}")


def verify_generated_xml(expected_mass):
    """Confirm that the generated XML contains the calculated mass."""
    print("\n[5/7] Verifying generated XML...")

    tree = ET.parse(OUTPUT_XML)
    mass_element = find_first_xml_element(tree.getroot(), "mass")

    if mass_element is None or mass_element.text is None:
        raise RuntimeError("The generated XML does not contain a mass value.")

    generated_mass = float(mass_element.text)

    if not np.isclose(generated_mass, expected_mass, rtol=0.0, atol=1e-6):
        raise RuntimeError(
            "Generated OpenSim XML mass does not match the Python-calculated mass."
        )

    print(f"XML verified successfully: {generated_mass:.6f} kg")


# ============================================================
# OPENSIM AUTOMATION
# ============================================================

def resolve_opensim_paths():
    """Return OpenSim command-line and GUI executable paths."""
    install_root = Path(os.environ.get("OPENSIM_HOME", r"C:\OpenSim 4.4"))

    cmd_candidates = [
        install_root / "bin" / "opensim-cmd.exe",
        Path(r"C:\OpenSim 4.4\bin\opensim-cmd.exe"),
        Path(r"C:\OpenSim 4.6\bin\opensim-cmd.exe"),
    ]
    gui_candidates = [
        install_root / "bin" / "OpenSim64.exe",
        Path(r"C:\OpenSim 4.4\bin\OpenSim64.exe"),
        Path(r"C:\OpenSim 4.6\bin\OpenSim64.exe"),
    ]

    cli = next((path for path in cmd_candidates if path.is_file()), None)
    gui = next((path for path in gui_candidates if path.is_file()), None)

    if cli is None and gui is None:
        searched = "\n".join(str(p) for p in cmd_candidates + gui_candidates)
        raise FileNotFoundError(
            "OpenSim executable not found. Checked:\n"
            f"{searched}\n"
            "Set OPENSIM_HOME to the OpenSim installation folder."
        )

    if cli is None:
        cli = gui.parent / "opensim-cmd.exe"
    if gui is None:
        gui = cli.parent / "OpenSim64.exe"

    return cli, gui


def run_opensim(opensim_cmd):
    """Run the OpenSim ScaleTool with the generated setup XML."""
    print("\n[6/7] Running OpenSim ScaleTool...")

    if not opensim_cmd.is_file():
        raise FileNotFoundError(f"OpenSim CLI not found: {opensim_cmd}")
    if not ORIGINAL_MODEL.is_file():
        raise FileNotFoundError(f"Model not found: {ORIGINAL_MODEL}")
    if not OUTPUT_XML.is_file():
        raise FileNotFoundError(f"XML not found: {OUTPUT_XML}")

    result = subprocess.run(
        [str(opensim_cmd), "run-tool", OUTPUT_XML.name],
        cwd=str(PROJECT_FOLDER),
        capture_output=True,
        text=True,
        errors="replace",
    )

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        if not SCALED_MODEL.is_file():
            raise RuntimeError(
                f"OpenSim returned exit code {result.returncode} and did not "
                "generate the scaled model."
            )
        print(
            f"WARNING: OpenSim returned exit code {result.returncode}, but "
            "the scaled model file exists."
        )


def visualize_model(opensim_gui):
    """Launch OpenSim GUI and ask PowerShell to open the scaled model."""
    print("\n[7/7] Opening scaled model in OpenSim GUI...")

    gui_loader = PROJECT_FOLDER / "open_model_in_gui.ps1"

    if not SCALED_MODEL.is_file():
        raise FileNotFoundError(f"Scaled model not found: {SCALED_MODEL}")
    if not opensim_gui.is_file():
        raise FileNotFoundError(f"OpenSim GUI not found: {opensim_gui}")
    if not gui_loader.is_file():
        raise FileNotFoundError(f"GUI loader not found: {gui_loader}")

    profile_folder = (
        Path(os.environ.get("TEMP", str(PROJECT_FOLDER)))
        / "OpenSim46_TestProfile"
    )

    gui_process = subprocess.Popen(
        [str(opensim_gui), "--userdir", str(profile_folder)],
        cwd=str(PROJECT_FOLDER),
    )

    subprocess.Popen(
        [
            "powershell.exe",
            "-NoProfile",
            "-STA",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(gui_loader),
            "-ProcessId",
            str(gui_process.pid),
            "-ModelPath",
            str(SCALED_MODEL),
        ],
        cwd=str(PROJECT_FOLDER),
        creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
    )

    print("OpenSim GUI launched; automatic model-loading request started.")
    print(f"Model: {SCALED_MODEL}")
    print("If the model does not load, use File > Open Model in OpenSim.")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    try:
        segments, total_volume, total_mass = calculate_body_measurements()
        write_measurement_report(segments, total_volume, total_mass)
        generate_opensim_xml(total_mass)
        verify_generated_xml(total_mass)

        opensim_cmd, opensim_gui = resolve_opensim_paths()
        run_opensim(opensim_cmd)
        visualize_model(opensim_gui)

        print("\n" + "=" * 75)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 75)
        print("Point cloud -> capped segment meshes -> divergence theorem")
        print(f"Total mass = {total_mass:.2f} kg")
        print(f"OpenSim setup = {OUTPUT_XML.name}")
        print(f"Scaled model = {SCALED_MODEL.name}")

    except Exception as error:
        print("\n" + "=" * 75)
        print("PIPELINE FAILED")
        print("=" * 75)
        print(f"\nERROR: {error}")
        raise


if __name__ == "__main__":
    main()
