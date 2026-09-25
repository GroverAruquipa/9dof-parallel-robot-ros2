# Nota de diseño: torsión casi libre de la plataforma 2

Resultado encontrado al simular en MuJoCo la geometría medida del CAD
(`model_6v3`, `config/geometry.yaml`). **Es preliminar**: conviene
confirmarlo con el modelo completo antes de construir el prototipo.

## Qué se observa

1. **Jacobiano casi singular en home.** El valor singular mínimo de K⁻¹J es
   ≈ 1·10⁻⁴ (κ ≈ 1400 con rotaciones normalizadas por 40 mm). La dirección
   peor condicionada es sobre todo la torsión β₃ de la plataforma 2.
2. **Casi-movilidad con los actuadores bloqueados (q = 0).** Resolviendo la
   cinemática exacta, la plataforma 2 puede girar 60° en β₃ con un error de
   longitud de las barras < 0,1 mm:

   | β₃ | error máx. de longitud |
   |---|---|
   | 0° | 0,6 µm |
   | 20° | 29 µm |
   | 60° | 95 µm |

3. **Dos modos de ensamblaje para q = 0**, ambos con residuo nulo: home y
   otro con la plataforma 2 girada ≈ 265° y ≈ 10 mm más abajo. En MuJoCo, con
   articulaciones ideales (error < 5 µm), el robot se mantiene en home ~1 s y
   luego salta al otro modo bajo su propio peso.

## Por qué

Las barras distales son casi verticales (inclinación media ≈ 8°) y su
pequeña inclinación no es tangencial, así que una rotación de la plataforma
alrededor del eje vertical casi no cambia su longitud.

Estudio paramétrico (mismas plataformas, κ con rotaciones normalizadas):

| Cambio de geometría | κ en home |
|---|---|
| CAD actual | ≈ 1400 |
| Punto de la base alineado radialmente con Aᵢ | singular |
| Aumentar r_b de 50 a 100 mm | ≈ 760 |
| Mismo giro tangencial en todas las patas | ≈ 10⁶ |
| **Giro tangencial alternado (patas cruzadas por pares), r_b = 70 mm, ±45°** | **≈ 85** |

## Recomendación

Disponer las patas en **pares cruzados** (como en una plataforma de
Gough-Stewart), sobre todo las 4 patas de la plataforma 2. El análisis se
puede repetir con cualquier geometría modificando `config/geometry.yaml`
y usando `ninedof_kinematics.kinematics.NineDofKinematics.jacobians`.
