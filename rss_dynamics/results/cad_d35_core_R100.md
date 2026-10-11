# Cotas para CAD: diseño `d35_core_R100`

Manivela d = 35.0 mm (centro del eje a centro de rótula), biela l = 148.38 mm (centro a centro de rótulas), centro de la rótula central O_p a z = 136.26 mm, 143.21 mm sobre los ejes de manivela.
Correa GT2: polea motor 16T, polea de manivela 64T (diámetro primitivo 40.7 mm).

Origen en el eje vertical de la base, z hacia arriba; z = 0 es el plano de referencia del diseño (los ejes de manivela están todos a la misma altura c_z). En home las dos plataformas tienen Q = I (pinza cerrada, como el CAD de graspability).

| pata | plataforma | eje de manivela c_i (mm) | dirección del eje v_i | ángulo de manivela en home: dirección d_i/|d_i| | punta b_i (centro rótula) (mm) | anclaje A_i en home (mm) | a_i en la plataforma (mm) | eje del asiento de rótula de plataforma (marco de plataforma) |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | (   44.99,    46.92,    -6.95) | (    0.77,    -0.64,     0.00) | (    0.52,     0.61,     0.60) | (   63.04,    68.42,    13.94) | (  -13.83,    34.55,   136.26) | (  -13.83,    34.55,     0.00) | (   -0.51,    -0.13,     0.85) |
| 2 | 1 | (   20.30,    61.75,    -6.95) | (    0.97,    -0.25,     0.00) | (    0.20,     0.79,     0.58) | (   27.42,    89.46,    13.20) | (  -26.82,    26.75,   136.26) | (  -26.82,    26.75,     0.00) | (   -0.37,    -0.32,     0.87) |
| 3 | 1 | (  -33.68,    55.60,    -6.95) | (    0.82,     0.57,    -0.00) | (   -0.54,     0.76,     0.36) | (  -52.42,    82.37,     5.58) | (  -35.29,    14.19,   136.26) | (  -35.29,    14.19,     0.00) | (    0.10,    -0.37,     0.92) |
| 4 | 1 | (  -43.32,    48.46,    -6.95) | (    0.70,     0.71,    -0.00) | (   -0.64,     0.62,     0.46) | (  -65.57,    70.26,     9.01) | (  -37.66,    -0.77,   136.26) | (  -37.66,    -0.77,    -0.00) | (    0.17,    -0.39,     0.90) |
| 5 | 1 | (  -64.99,    -1.37,    -6.95) | (   -0.09,     1.00,     0.00) | (   -0.95,    -0.08,     0.29) | (  -98.34,    -4.28,     3.25) | (  -33.49,   -15.34,   136.26) | (  -33.49,   -15.34,     0.00) | (    0.42,    -0.00,     0.91) |
| 6 | 2 | (   64.23,    -9.95,    -6.95) | (   -0.09,    -1.00,     0.00) | (    0.97,    -0.09,     0.23) | (   98.17,   -12.93,     1.07) | (   37.61,    -4.31,   136.26) | (   37.61,    -4.31,    -0.00) | (   -0.40,    -0.02,     0.92) |
| 7 | 2 | (   64.97,     2.04,    -6.95) | (    0.10,    -1.00,     0.00) | (    0.95,     0.09,     0.29) | (   98.32,     5.30,     3.16) | (   33.49,    15.34,   136.26) | (   33.49,    15.34,     0.00) | (   -0.43,    -0.00,     0.90) |
| 8 | 2 | (  -26.23,   -59.47,    -6.95) | (   -0.94,     0.34,     0.00) | (   -0.31,    -0.85,     0.43) | (  -37.06,   -89.19,     8.03) | (   13.83,   -34.55,   136.26) | (   13.83,   -34.55,     0.00) | (    0.34,     0.27,     0.90) |
| 9 | 2 | (    2.04,   -64.97,    -6.95) | (   -1.00,    -0.10,     0.00) | (    0.09,    -0.88,     0.47) | (    5.05,   -95.76,     9.42) | (   30.25,   -22.99,   136.26) | (   30.25,   -22.99,     0.00) | (    0.17,     0.39,     0.90) |

| pata | radio del eje (mm) | azimut del eje (deg) | elevación de la manivela en home (deg) | radio de la punta (mm) |
|---|---|---|---|---|
| 1 | 65.00 | -40.01 | 36.65 | 93.04 |
| 2 | 65.00 | -14.41 | 35.16 | 93.57 |
| 3 | 65.00 | 34.99 | 20.98 | 97.63 |
| 4 | 65.00 | 45.59 | 27.12 | 96.11 |
| 5 | 65.00 | 94.99 | 16.95 | 98.43 |
| 6 | 65.00 | -95.02 | 13.25 | 99.02 |
| 7 | 65.00 | -84.42 | 16.80 | 98.46 |
| 8 | 65.00 | 159.99 | 25.35 | 96.58 |
| 9 | 65.00 | -174.42 | 27.89 | 95.89 |

Radio barrido por las manivelas: 100.0 mm. Altura de los ejes: z = -6.95 mm.
Recorrido máximo de manivela en el showcase: ±76 deg; desalineación del rod-end: 26 deg; cono de rótula de plataforma: 39 deg.
