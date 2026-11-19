PATIENT-SPECIFIC SCALING: STEPS 1-6
===================================
Open a terminal in this New Files folder.

INSTALL PYTHON DEPENDENCIES:
  py -m pip install numpy scipy

TEST WITHOUT OPENSIM (recommended first):
  py main.py --dry-run

CHECK THESE GENERATED FILES:
  subject_measurements.txt        patient geometric measurements in mm
  P2_Scaled_ScaleSetup.xml        generated OpenSim manualScale XML

FULL RUN (Windows with OpenSim installed):
  py main.py

If OpenSim is installed elsewhere, set OPENSIM_HOME to the install folder:
  $env:OPENSIM_HOME = 'C:\OpenSim 4.6'
  py main.py

EXPECTED DRY-RUN OUTPUT FOR THE BUNDLED SAMPLE:
  Patient point-cloud height: ~1713.6 mm
  Left arm ratio:  ~0.93478
  Right arm ratio: ~0.91671
  Left leg ratio:  ~0.89965
  Right leg ratio: ~0.89454
  All 20 limb scaling entries verified.

IMPORTANT LIMITATIONS:
- Reference model is P2_APP1_SO.osim; mesh dimensions come from the
  original model's linked Geometry/*.vtp files.
- The model references some mesh filenames that differ from the actual
  shipped filenames; the loader resolves unique filename-prefix matches.
- Arm ratios compare the scan's entire arm principal-axis extent with the
  sum of humerus/ulna/hand mesh lengths. Leg ratios compare the scan's
  leg principal-axis extent with femur/tibia mesh lengths. These are
  APPROXIMATIONS, not clinically validated anatomical joint distances.
- Torso, pelvis, head, and patient height are recorded, but the program
  does NOT apply ungrounded scale ratios to these body segments. The
  template's pelvis scaling remains as originally specified.
- The original template is not overwritten. The generated output XML
  is the file run_opensim() uses.
- This project estimates body mass using convex hulls and uniform density;
  overlapping segment hulls can overestimate mass. Validate independently.
- The dry run verifies Python and XML, not OpenSim's runtime. Only a full
  run with a local OpenSim installation validates OpenSim scaling and GUI.
