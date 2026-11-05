import os
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.io import loadmat
from scipy.spatial import ConvexHull
from scipy.spatial import QhullError


# ============================================================
# CONFIGURATION
# ============================================================

# Folder containing this Python script
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

# Optional measurement report
OUTPUT_TXT = PROJECT_FOLDER / "subject_measurements.txt"

# Tissue density:
# 1000 kg/m^3 = 1e-6 kg/mm^3
DENSITY_KG_PER_MM3 = 1e-6


# ============================================================
# LOAD A VARIABLE FROM A MATLAB FILE
# ============================================================

def load_mat_variable(file_path, variable_name):
    """
    Loads one variable from a standard MATLAB .mat file.

    For example:
        points = load_mat_variable("points.mat", "points")
    """

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
            "Resave the file in MATLAB using the standard MAT format, "
            "or use the mat73 package."
        ) from error

    if variable_name not in mat_data:
        available_variables = [
            key
            for key in mat_data.keys()
            if not key.startswith("__")
        ]

        raise KeyError(
            f'The variable "{variable_name}" was not found in '
            f"{file_path.name}.\n"
            f"Available variables: {available_variables}"
        )

    return np.asarray(mat_data[variable_name])


# ============================================================
# VALIDATE POINT-CLOUD INPUTS
# ============================================================

def validate_inputs(points, faces):
    """
    Validates the main point and face matrices.
    """

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            "The points variable must be an N x 3 matrix containing "
            "X, Y, and Z coordinates."
        )

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            "The faces variable must be an M x 3 matrix containing "
            "three vertex indices per face."
        )

    if not np.all(np.isfinite(points)):
        raise ValueError(
            "The point cloud contains NaN or infinite values."
        )


# ============================================================
# CONVERT MATLAB INDICES TO PYTHON INDICES
# ============================================================

def convert_matlab_indices(index_array, maximum_size, index_name):
    """
    MATLAB arrays use one-based indexing:
        first item = 1

    Python arrays use zero-based indexing:
        first item = 0

    This function converts MATLAB indices to Python indices.
    """

    indices = np.asarray(index_array).reshape(-1)

    if indices.size == 0:
        raise ValueError(f"{index_name} contains no indices.")

    if not np.all(np.isfinite(indices)):
        raise ValueError(
            f"{index_name} contains invalid values."
        )

    # Convert values to integers
    indices = indices.astype(np.int64)

    # Validate them as MATLAB indices before subtracting one
    if np.any(indices < 1) or np.any(indices > maximum_size):
        raise IndexError(
            f"{index_name} contains values outside the valid "
            f"MATLAB range of 1 to {maximum_size}."
        )

    # MATLAB index -> Python index
    return indices - 1


# ============================================================
# CALCULATE ONE BODY SEGMENT
# ============================================================

def calculate_segment(
    segment_name,
    points,
    faces,
    matlab_segment_face_indices
):
    """
    Reproduces the MATLAB process:

        segment vertex indices = unique(faces(segment faces, :))
        segment points = points(segment vertex indices, :)
        length = maximum Z - minimum Z
        volume = 3D convex-hull volume
        mass = volume x density
    """

    # Convert the segment's face indices from MATLAB to Python
    segment_face_indices = convert_matlab_indices(
        matlab_segment_face_indices,
        len(faces),
        f"{segment_name} face indices"
    )

    # Get the three point indices belonging to every selected face
    selected_faces = faces[segment_face_indices]

    # faces itself also uses MATLAB one-based point indexing
    point_indices = np.unique(selected_faces.reshape(-1))

    if np.any(point_indices < 1) or np.any(
        point_indices > len(points)
    ):
        raise IndexError(
            f"{segment_name} contains invalid point indices."
        )

    # Convert point indices from MATLAB to Python
    point_indices_python = point_indices.astype(np.int64) - 1

    # Extract XYZ coordinates
    segment_points = points[point_indices_python]

    if len(segment_points) < 4:
        raise ValueError(
            f"{segment_name} has fewer than four points, so a "
            "3D convex hull cannot be calculated."
        )

    # Same length calculation as MATLAB:
    # max Z minus min Z
    segment_length_mm = (
        np.max(segment_points[:, 2])
        - np.min(segment_points[:, 2])
    )

    # Same convex-hull volume approach as MATLAB
    try:
        hull = ConvexHull(segment_points)
        segment_volume_mm3 = float(hull.volume)

    except QhullError as error:
        raise RuntimeError(
            f"Convex-hull calculation failed for {segment_name}."
        ) from error

    # Convert volume to estimated mass
    segment_mass_kg = (
        segment_volume_mm3 * DENSITY_KG_PER_MM3
    )

    if segment_length_mm <= 0:
        raise ValueError(
            f"{segment_name} produced an invalid length."
        )

    if segment_volume_mm3 <= 0:
        raise ValueError(
            f"{segment_name} produced an invalid volume."
        )

    if segment_mass_kg <= 0:
        raise ValueError(
            f"{segment_name} produced an invalid mass."
        )

    return {
        "name": segment_name,
        "point_count": len(segment_points),
        "length_mm": float(segment_length_mm),
        "volume_mm3": segment_volume_mm3,
        "mass_kg": segment_mass_kg
    }


# ============================================================
# LOAD ALL DATA AND CALCULATE BODY MEASUREMENTS
# ============================================================

def calculate_body_measurements():
    """
    Loads the point-cloud files and calculates all segment values.
    """

    print("=" * 65)
    print("FULLY AUTOMATED PYTHON-OPENSIM PIPELINE")
    print("=" * 65)

    print("\n[1/5] Loading point-cloud data...")

    points = load_mat_variable(
        POINTS_FILE,
        "points"
    ).astype(float)

    faces = load_mat_variable(
        FACES_FILE,
        "faces"
    ).astype(np.int64)

    idx_la = load_mat_variable(
        LEFT_ARM_INDEX_FILE,
        "idxLA"
    )

    idx_ra = load_mat_variable(
        RIGHT_ARM_INDEX_FILE,
        "idxRA"
    )

    idx_ll = load_mat_variable(
        LEFT_LEG_INDEX_FILE,
        "idxLL"
    )

    idx_rl = load_mat_variable(
        RIGHT_LEG_INDEX_FILE,
        "idxRL"
    )

    idx_head = load_mat_variable(
        HEAD_INDEX_FILE,
        "idxHead"
    )

    total_idx = load_mat_variable(
        TRUNK_INDEX_FILE,
        "TotalIdx"
    )

    validate_inputs(points, faces)

    print(f"Loaded {len(points):,} point-cloud vertices.")
    print(f"Loaded {len(faces):,} triangular faces.")

    print("\n[2/5] Calculating segment properties...")

    segments = {
        "LEFT_ARM": calculate_segment(
            "Left Arm",
            points,
            faces,
            idx_la
        ),
        "RIGHT_ARM": calculate_segment(
            "Right Arm",
            points,
            faces,
            idx_ra
        ),
        "LEFT_LEG": calculate_segment(
            "Left Leg",
            points,
            faces,
            idx_ll
        ),
        "RIGHT_LEG": calculate_segment(
            "Right Leg",
            points,
            faces,
            idx_rl
        ),
        "HEAD": calculate_segment(
            "Head",
            points,
            faces,
            idx_head
        ),
        "TRUNK": calculate_segment(
            "Trunk",
            points,
            faces,
            total_idx
        )
    }

    total_mass = sum(
        segment["mass_kg"]
        for segment in segments.values()
    )

    total_volume = sum(
        segment["volume_mm3"]
        for segment in segments.values()
    )

    if not np.isfinite(total_mass) or total_mass <= 0:
        raise ValueError("Calculated total mass is invalid.")

    print("\nSegment results:")

    for segment in segments.values():
        print(
            f"  {segment['name']:<10} "
            f"{segment['mass_kg']:>7.2f} kg | "
            f"Length: {segment['length_mm']:>7.1f} mm | "
            f"Volume: {segment['volume_mm3']:.2f} mm^3"
        )

    print("-" * 65)
    print(f"TOTAL VOLUME: {total_volume:.2f} mm^3")
    print(f"TOTAL MASS:   {total_mass:.2f} kg")

    return segments, total_volume, total_mass


# ============================================================
# WRITE MEASUREMENT REPORT
# ============================================================

def write_measurement_report(
    segments,
    total_volume,
    total_mass
):
    """
    Writes a text report for debugging and record keeping.

    OpenSim does not read this text file. The XML receives the
    Python-calculated mass directly.
    """

    print("\n[3/5] Writing measurement report...")

    with OUTPUT_TXT.open("w", encoding="utf-8") as file:
        file.write("STATUS=SUCCESS\n")
        file.write("SOURCE=PYTHON_POINT_CLOUD_PIPELINE\n")
        file.write("MASS_UNIT=kg\n")
        file.write("LENGTH_UNIT=mm\n")
        file.write("VOLUME_UNIT=mm3\n")
        file.write("VOLUME_METHOD=CONVEX_HULL\n")
        file.write("LENGTH_METHOD=Z_AXIS_RANGE\n")

        file.write(
            f"DENSITY_KG_PER_MM3="
            f"{DENSITY_KG_PER_MM3:.10f}\n"
        )

        file.write(f"TOTAL_MASS={total_mass:.6f}\n")
        file.write(f"TOTAL_VOLUME={total_volume:.6f}\n")

        for key, segment in segments.items():
            file.write(
                f"{key}_MASS="
                f"{segment['mass_kg']:.6f}\n"
            )

            file.write(
                f"{key}_LENGTH="
                f"{segment['length_mm']:.6f}\n"
            )

            file.write(
                f"{key}_VOLUME="
                f"{segment['volume_mm3']:.6f}\n"
            )

            file.write(
                f"{key}_POINT_COUNT="
                f"{segment['point_count']}\n"
            )

    print(f"Generated: {OUTPUT_TXT.name}")


# ============================================================
# UPDATE OPENSIM SCALE XML
# ============================================================

def generate_opensim_xml(total_mass):
    """
    Inserts the Python-calculated total mass directly into the
    OpenSim ScaleTool XML template.
    """

    print("\n[4/5] Generating OpenSim XML...")

    if not TEMPLATE_XML.exists():
        raise FileNotFoundError(
            f"OpenSim template was not found:\n{TEMPLATE_XML}"
        )

    tree = ET.parse(TEMPLATE_XML)
    root = tree.getroot()

    mass_element = None

    for element in root.iter():
        tag_name = element.tag.split("}")[-1]

        if tag_name == "mass":
            mass_element = element
            break

    if mass_element is None:
        raise RuntimeError(
            "No <mass> element was found in the OpenSim XML."
        )

    previous_mass = mass_element.text

    mass_element.text = f"{total_mass:.6f}"

    if hasattr(ET, "indent"):
        ET.indent(tree, space="\t", level=0)

    tree.write(
        OUTPUT_XML,
        encoding="UTF-8",
        xml_declaration=True
    )

    print(f"Old XML mass: {previous_mass} kg")
    print(f"New XML mass: {total_mass:.6f} kg")
    print(f"Generated: {OUTPUT_XML.name}")


# ============================================================
# VERIFY XML
# ============================================================

def verify_generated_xml(expected_mass):
    """
    Reopens the generated XML and confirms that its mass matches
    the mass calculated from the point cloud.
    """

    print("\n[5/5] Verifying generated XML...")

    tree = ET.parse(OUTPUT_XML)
    root = tree.getroot()

    generated_mass = None

    for element in root.iter():
        tag_name = element.tag.split("}")[-1]

        if tag_name == "mass":
            generated_mass = float(element.text)
            break

    if generated_mass is None:
        raise RuntimeError(
            "The generated XML does not contain a mass value."
        )

    if not np.isclose(
        generated_mass,
        expected_mass,
        rtol=0.0,
        atol=1e-6
    ):
        raise RuntimeError(
            "Generated OpenSim XML mass does not match the "
            "Python-calculated mass."
        )

    print(
        f"XML verified successfully: "
        f"{generated_mass:.6f} kg"
    )
# ============================================================
# OPENSIM AUTOMATION
# ============================================================


def resolve_opensim_paths():
    """Return the CLI and GUI executables for the installed OpenSim build."""
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
            "Set OPENSIM_HOME to the OpenSim install folder or use the correct bin path."
        )

    if cli is None:
        cli = gui.parent / "opensim-cmd.exe"
    if gui is None:
        gui = cli.parent / "OpenSim64.exe"

    return cli, gui


OPENSIM_CMD, OPENSIM_GUI = resolve_opensim_paths()

ORIGINAL_MODEL = PROJECT_FOLDER / "P2_APP1_SO.osim"
SCALED_MODEL = PROJECT_FOLDER / "P2_PointCloud_scaled.osim"
GEOMETRY_FOLDER = PROJECT_FOLDER / "Geometry"


def run_opensim():
    print("\n[6/7] Running OpenSim ScaleTool...")

    if not OPENSIM_CMD.is_file():
        raise FileNotFoundError(f"OpenSim CLI not found: {OPENSIM_CMD}")

    if not ORIGINAL_MODEL.is_file():
        raise FileNotFoundError(f"Model not found: {ORIGINAL_MODEL}")

    if not OUTPUT_XML.is_file():
        raise FileNotFoundError(f"XML not found: {OUTPUT_XML}")

    command = [
        str(OPENSIM_CMD),
        "run-tool",
        OUTPUT_XML.name
    ]

    result = subprocess.run(
        command,
        cwd=str(PROJECT_FOLDER),
        capture_output=True,
        text=True,
        errors="replace"
    )

    print(result.stdout)
    print(result.stderr)

    if result.returncode != 0:
        print(
            f"WARNING: OpenSim returned exit code {result.returncode}."
        )

        if not SCALED_MODEL.is_file():
            raise RuntimeError(
                "OpenSim failed and did not generate the scaled model."
            )

        print(
            "The scaled model file exists, but OpenSim reported an error."
        )

# ============================================================
# MAIN PIPELINE
# ============================================================
def visualize_model():
    """Launch the OpenSim desktop GUI and request that it open the scaled model."""
    print("\n[7/7] Opening scaled model in OpenSim GUI...")

    opensim_gui = OPENSIM_GUI
    gui_loader = PROJECT_FOLDER / "open_model_in_gui.ps1"

    if not SCALED_MODEL.is_file():
        raise FileNotFoundError(f"Scaled model not found: {SCALED_MODEL}")
    if not opensim_gui.is_file():
        raise FileNotFoundError(f"OpenSim GUI not found: {opensim_gui}")
    if not gui_loader.is_file():
        raise FileNotFoundError(f"GUI loader not found: {gui_loader}")

    import os
    profile_folder = Path(os.environ.get("TEMP", str(PROJECT_FOLDER))) / "OpenSim46_TestProfile"
    gui_process = subprocess.Popen(
        [str(opensim_gui), "--userdir", str(profile_folder)],
        cwd=str(PROJECT_FOLDER)
    )

    # Windows GUI automation runs separately so the Python pipeline can finish.
    # It waits for OpenSim's window, then uses File > Open Model (Ctrl+O).
    subprocess.Popen(
        [
            "powershell.exe", "-NoProfile", "-STA", "-ExecutionPolicy", "Bypass",
            "-File", str(gui_loader),
            "-ProcessId", str(gui_process.pid),
            "-ModelPath", str(SCALED_MODEL)
        ],
        cwd=str(PROJECT_FOLDER),
        creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    )
    print("OpenSim GUI launched; automatic model-loading request started.")
    print(f"Model: {SCALED_MODEL}")
    print("If the model is not loaded, use File > Open Model in OpenSim.")


def main():
    try:
        # Point cloud -> Python calculations
        segments, total_volume, total_mass = (
            calculate_body_measurements()
        )

        # Optional record of calculations
        write_measurement_report(
            segments,
            total_volume,
            total_mass
        )

        # Python value -> OpenSim XML
        generate_opensim_xml(total_mass)

        # Confirm output
        verify_generated_xml(total_mass)

        # Run OpenSim scaling
        run_opensim()

        # Open the scaled model automatically
        visualize_model()

        print("\n" + "=" * 65)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 65)

        print("\nPoint cloud")
        print("    |")
        print("    v")
        print("Python segment calculations")
        print("    |")
        print("    v")
        print(f"Total mass = {total_mass:.2f} kg")
        print("    |")
        print("    v")
        print(OUTPUT_XML.name)

    except Exception as error:
        print("\n" + "=" * 65)
        print("PIPELINE FAILED")
        print("=" * 65)
        print(f"\nERROR: {error}")
        raise


if __name__ == "__main__":
    main()