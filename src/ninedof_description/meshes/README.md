# Meshes

Exported from SolidWorks (`model_6v3`) in meters, then re-centred so that each
mesh origin sits on its joint:

| File | CAD part | Origin | Notes |
|---|---|---|---|
| `base.stl` | `base)robot` | Centre of the bottom face, Z up | Ring Ø120/Ø80 mm, 30 mm high; 10 actuator holes Ø5 mm at r = 50 mm |
| `slider.stl` | `arm2` | Bottom of the rod, Z along the actuator | Rod Ø5 mm; ball B_i (R 5 mm) centre at z = 50.5 mm |
| `distal_link.stl` | `arm1` | Centre of the lower ball B_i, Z towards A_i | 120 mm between ball centres |
| `platform_1.stl` | `platform_1a` + `p1` | Centre of the central spherical joint | 5 ball joints A_i (R 4 mm) at r ≈ 37.5 mm; rotated 180° about Z after export, then by −116.5° so that Q1 = I at home |
| `platform_2.stl` | `platform_1b` + `p1` | Centre of the central spherical joint | 4 ball joints A_i (R 4 mm) at r ≈ 37.5 mm; rotated by −116.5° so that Q2 = I at home |

The leg geometry (`config/geometry.yaml`) was measured on the full assembly
export `hardware/model_6v3_assembly.stl`: every distal link measures 120.00 mm
between ball centres, and the base hole at −15° is not used.
