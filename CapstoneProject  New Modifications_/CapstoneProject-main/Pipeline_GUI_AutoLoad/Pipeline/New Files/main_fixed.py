import os
import subprocess
import argparse
import json
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

    # Principal-axis extent is orientation-independent for limbs.
    # It is a geometric estimate, not a joint-to-joint anatomical length.
    centered = segment_points - np.mean(segment_points, axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ axes[0]
    principal_length_mm = float(np.ptp(projected))
    z_extent_mm = float(np.ptp(segment_points[:, 2]))
    x_extent_mm = float(np.ptp(segment_points[:, 0]))
    y_extent_mm = float(np.ptp(segment_points[:, 1]))

    # For the head/trunk use the vertical extent; for limbs use the
    # principal-axis extent, which also works for nonvertical limbs.
    segment_length_mm = (z_extent_mm if segment_name in ("Head", "Trunk")
                         else principal_length_mm)

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
        "principal_length_mm": principal_length_mm,
        "height_mm": z_extent_mm,
        "width_mm": x_extent_mm,
        "depth_mm": y_extent_mm,
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

    # Assumes Z is the vertical axis and the scan contains the whole body.
    # This is overall point-cloud height, not a calibrated standing height
    # if the patient is bent, seated, or incompletely scanned.
    patient_height_mm = float(np.ptp(points[:, 2]))
    if patient_height_mm <= 0:
        raise ValueError("Invalid patient point-cloud height")
    print(f"POINT-CLOUD HEIGHT: {patient_height_mm:.1f} mm")
    return segments, total_volume, total_mass, patient_height_mm


# ============================================================
# WRITE MEASUREMENT REPORT
# ============================================================

def write_measurement_report(
    segments,
    total_volume,
    total_mass,
    patient_height_mm
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
        file.write("LENGTH_METHOD=PRINCIPAL_AXIS_FOR_LIMBS_Z_FOR_HEAD_TRUNK\n")
        file.write("COORDINATE_ASSUMPTION=Z_UP_X_WIDTH_Y_DEPTH\n")
        file.write("MEASUREMENT_TYPE=GEOMETRIC_ESTIMATE_NOT_JOINT_LANDMARKS\n")
        file.write(f"PATIENT_HEIGHT_MM={patient_height_mm:.6f}\n")
        file.write(f"PATIENT_HEIGHT_M={patient_height_mm / 1000:.6f}\n")

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

            for field, label in (
                ("principal_length_mm", "PRINCIPAL_LENGTH_MM"),
                ("height_mm", "HEIGHT_MM"),
                ("width_mm", "WIDTH_MM"),
                ("depth_mm", "DEPTH_MM"),
            ):
                file.write(f"{key}_{label}={segment[field]:.6f}\n")

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

def read_measurements(path=OUTPUT_TXT):
    """Step 2: parse and validate the saved patient measurements, not in-memory values."""
    values = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Malformed measurement line: {line}")
        key, value = line.split("=", 1)
        if key in values:
            raise ValueError(f"Duplicate measurement key: {key}")
        values[key] = value.strip()
    if values.get("STATUS") != "SUCCESS" or values.get("LENGTH_UNIT") != "mm":
        raise ValueError("Measurements must be successful and in millimetres")
    for key in ("PATIENT_HEIGHT_MM", "TOTAL_MASS", "LEFT_ARM_LENGTH",
                "RIGHT_ARM_LENGTH", "LEFT_LEG_LENGTH", "RIGHT_LEG_LENGTH",
                "TRUNK_HEIGHT_MM", "TRUNK_WIDTH_MM", "HEAD_HEIGHT_MM"):
        value = float(values[key])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid measurement: {key}={value}")
    return values


def model_reference_dimensions(model_path=ORIGINAL_MODEL if 'ORIGINAL_MODEL' in globals() else PROJECT_FOLDER / 'P2_APP1_SO.osim'):
    """Step 3: reference dimensions from original model's joint offsets.

    These are joint-center lengths where available, with VTP mesh dimensions
    used for femurs (whose knee joint uses a custom transform) and body widths.
    No patient-to-model landmark correspondence is asserted.
    """
    root = ET.parse(model_path).getroot()
    bodies = {b.get("name"): b for b in root.iter("Body") if b.get("name")}
    def joint_offset(child):
        b = bodies[child]
        for joint in b.findall('./Joint/*'):
            loc = joint.findtext('location_in_parent')
            if loc:
                return float(np.linalg.norm(np.fromstring(loc, sep=' '))) * 1000
        raise ValueError(f"Missing joint offset for {child}")
    def mesh_points(body_name):
        body = bodies[body_name]
        points = []
        for geometry in body.findall('.//DisplayGeometry'):
            filename = geometry.findtext('geometry_file')
            if not filename:
                continue
            candidate = PROJECT_FOLDER / 'Geometry' / filename
            if not candidate.exists():
                # This project's .osim refers to femur_r.vtp but ships femur_rBruno.vtp.
                candidates = sorted((PROJECT_FOLDER / 'Geometry').glob(Path(filename).stem + '*'+Path(filename).suffix))
                if len(candidates) != 1:
                    continue
                candidate = candidates[0]
            try:
                vtproot = ET.parse(candidate).getroot()
                array = next(a for a in vtproot.findall('.//Points/DataArray') if a.get('format', 'ascii') == 'ascii')
                coords = np.fromstring(array.text or '', sep=' ').reshape(-1, 3)
                factors = np.fromstring(geometry.findtext('scale_factors', '1 1 1'), sep=' ')
                if len(factors) == 3:
                    coords = coords * factors
                points.append(coords)
            except (ET.ParseError, ValueError, StopIteration):
                continue
        if not points:
            raise ValueError(f"No readable ASCII VTP geometry for model body {body_name}")
        coords = np.vstack(points)
        # OpenSim Body VisibleObject's display scale factors.
        visible = body.find('VisibleObject')
        if visible is not None:
            factors = np.fromstring(visible.findtext('scale_factors', '1 1 1'), sep=' ')
            if len(factors) == 3:
                coords = coords * factors
        return coords * 1000  # model geometry in metres -> millimetres
    def mesh_long(body):
        pts = mesh_points(body)
        centered = pts - pts.mean(axis=0)
        _, _, v = np.linalg.svd(centered, full_matrices=False)
        return float(np.ptp(centered @ v[0]))
    def mesh_width(body):
        return float(np.ptp(mesh_points(body)[:, 0]))
    # Full-limb reference approximations: upper/lower bone geometry, not true
    # anatomical scan-to-model marker correspondence. No height-only fallback.
    dims = {
        'LEFT_ARM': mesh_long('humerus_L') + mesh_long('ulna_L') + mesh_long('hand_L'),
        'RIGHT_ARM': mesh_long('humerus_R') + mesh_long('ulna_R') + mesh_long('hand_R'),
        'LEFT_LEG': mesh_long('femur_l') + mesh_long('tibia_l'),
        'RIGHT_LEG': mesh_long('femur_r') + mesh_long('tibia_r'),
        'PELVIS_WIDTH': mesh_width('pelvis'),
    }
    # A trunk-wide scan measurement is not equivalent to pelvis width; avoid
    # fabricating a torso reference or scaling it by unrelated landmarks.
    for name, value in dims.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid model reference dimension {name}: {value}")
    return dims


def calculate_scale_factors(measurements, reference):
    """Step 4: ratio of point-cloud lengths to model mesh-derived lengths."""
    result = {}
    for segment in ('LEFT_ARM', 'RIGHT_ARM', 'LEFT_LEG', 'RIGHT_LEG'):
        ratio = float(measurements[segment + '_LENGTH']) / reference[segment]
        if not 0.5 <= ratio <= 2.0:
            raise ValueError(f"Implausible {segment} scale {ratio:.3f}; check units and geometry")
        result[segment] = ratio
    return result


SCALE_BODIES = {
    'LEFT_ARM': ('humerus_L', 'ulna_L', 'radius_L', 'hand_L'),
    'RIGHT_ARM': ('humerus_R', 'ulna_R', 'radius_R', 'hand_R'),
    'LEFT_LEG': ('femur_l', 'patella_l', 'tibia_l', 'talus_l', 'calcn_l', 'toes_l'),
    'RIGHT_LEG': ('femur_r', 'patella_r', 'tibia_r', 'talus_r', 'calcn_r', 'toes_r'),
}


def generate_opensim_xml(measurements, factors):
    """Step 5: write actual computed scale ratios into manualScale ScaleSet."""
    tree = ET.parse(TEMPLATE_XML)
    root = tree.getroot()
    scale_tool = root.find('.//ScaleTool')
    if scale_tool is None:
        raise ValueError('ScaleTool missing from template')
    scale_tool.find('mass').text = f"{float(measurements['TOTAL_MASS']):.6f}"
    scale_tool.find('height').text = f"{float(measurements['PATIENT_HEIGHT_MM']):.6f}"
    model_scaler = scale_tool.find('ModelScaler')
    if model_scaler is None:
        raise ValueError('ModelScaler missing from template')
    model_scaler.find('scaling_order').text = 'manualScale'
    objects = model_scaler.find('./ScaleSet/objects')
    if objects is None:
        raise ValueError('ScaleSet/objects missing from template')
    model_bodies = {b.get('name') for b in ET.parse(ORIGINAL_MODEL).getroot().iter('Body')}
    # Preserve the template's pelvis scale; replace any prior limb entries.
    limb_bodies = {name for names in SCALE_BODIES.values() for name in names}
    for scale in list(objects.findall('Scale')):
        if scale.findtext('segment') in limb_bodies:
            objects.remove(scale)
    for group, names in SCALE_BODIES.items():
        for name in names:
            if name not in model_bodies:
                raise ValueError(f"Scale target missing in base model: {name}")
            scale = ET.SubElement(objects, 'Scale')
            ET.SubElement(scale, 'scales').text = ' '.join([f'{factors[group]:.8f}'] * 3)
            ET.SubElement(scale, 'segment').text = name
            ET.SubElement(scale, 'apply').text = 'true'
    if hasattr(ET, 'indent'):
        ET.indent(tree, space='\t')
    tree.write(OUTPUT_XML, encoding='UTF-8', xml_declaration=True)
    print(f"Generated: {OUTPUT_XML.name} with {len(limb_bodies)} limb scale entries")


# ============================================================
# VERIFY XML
# ============================================================

def verify_generated_xml(expected_mass, expected_factors=None):
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

    if expected_factors is not None:
        found = {x.findtext('segment'): x for x in root.findall('.//ModelScaler/ScaleSet/objects/Scale')}
        for group, names in SCALE_BODIES.items():
            for name in names:
                if name not in found:
                    raise ValueError(f"Missing scale entry: {name}")
                values = np.fromstring(found[name].findtext('scales', ''), sep=' ')
                if len(values) != 3 or not np.allclose(values, expected_factors[group], atol=1e-7):
                    raise ValueError(f"Scale verification failed for {name}")
        print('Verified all 20 computed limb scaling entries.')

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


OPENSIM_CMD = None
OPENSIM_GUI = None  # Resolve only when actually launching OpenSim.

ORIGINAL_MODEL = PROJECT_FOLDER / "P2_APP1_SO.osim"
SCALED_MODEL = PROJECT_FOLDER / "P2_PointCloud_scaled.osim"
GEOMETRY_FOLDER = PROJECT_FOLDER / "Geometry"


def run_opensim():
    global OPENSIM_CMD, OPENSIM_GUI
    OPENSIM_CMD, OPENSIM_GUI = resolve_opensim_paths()
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

    # Avoid mistaking a stale previously generated model for a new successful run.
    if SCALED_MODEL.exists():
        SCALED_MODEL.unlink()

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

        if SCALED_MODEL.stat().st_size == 0:
            raise RuntimeError(
                f"OpenSim created an empty scaled model: {SCALED_MODEL}"
            )

        print(
            "OpenSim returned a nonzero exit code, but it generated "
            "a non-empty scaled model. Continuing to visualization."
        )

    if not SCALED_MODEL.is_file():
        raise RuntimeError(
            f"Expected scaled model was not generated: {SCALED_MODEL}"
        )

    if SCALED_MODEL.stat().st_size == 0:
        raise RuntimeError(
            f"Generated scaled model is empty: {SCALED_MODEL}"
        )

    print(f"Scaled model generated successfully: {SCALED_MODEL}")

# ============================================================
# MAIN PIPELINE
# ============================================================
def visualize_model():
    global OPENSIM_CMD, OPENSIM_GUI
    if OPENSIM_GUI is None:
        OPENSIM_CMD, OPENSIM_GUI = resolve_opensim_paths()
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
    parser = argparse.ArgumentParser(description='Point-cloud patient-specific OpenSim scaling')
    parser.add_argument('--dry-run', action='store_true', help='Generate and validate files without launching OpenSim')
    args = parser.parse_args()
    try:
        # Point cloud -> Python calculations
        segments, total_volume, total_mass, patient_height_mm = (
            calculate_body_measurements()
        )

        # Optional record of calculations
        write_measurement_report(
            segments,
            total_volume,
            total_mass,
            patient_height_mm
        )

        # Step 2: deliberately reload the report from disk.
        measurements = read_measurements()
        # Steps 3-4: extract base-model geometry and compute patient/model ratios.
        reference = model_reference_dimensions(ORIGINAL_MODEL)
        factors = calculate_scale_factors(measurements, reference)
        print("Model reference dimensions (mm):", reference)
        print("Computed limb scale factors:", factors)
        # Step 5: generate OpenSim manualScale XML.
        generate_opensim_xml(measurements, factors)

        # Confirm output
        verify_generated_xml(total_mass, factors)

        # Step 6: use the existing OpenSim runner unless dry-run was requested.
        if not args.dry_run:
            run_opensim()
            visualize_model()
        else:
            print('DRY RUN: OpenSim execution and GUI launch skipped.')

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