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

## Hallazgos al preparar la demo de pick-and-place (sin cambiar el CAD)

* **Montaje invertido.** Colgando (base arriba, pinza abajo) el modo casi libre
  queda en equilibrio estable: el robot ya no colapsa en reposo. De pie es un
  "péndulo invertido" y cambia de modo de ensamblaje en ~1 s.
* **La guiñada común de la pinza apenas se sostiene.** Con actuadores de
  5·10⁴ N/m, el error estático es ≤ 4° solo con ψ entre −10° y +25°; a
  ψ = ±45° llega a 7–14° y a −70° a ~40°. Para sostener ψ = −70° los
  actuadores saturan en 20 N aunque las plataformas pesan 25 g (la fuerza
  necesaria crece como 1/σ_min). Un resorte en la esférica central no ayuda:
  la deriva es una rotación común de ambas plataformas.
* **Dedos.** Los dedos son ganchos cortos simétricos por punto (uno es el otro
  girado 180°): agarran objetos de ~1 cm entre las puntas; un bloque de 40 mm
  no cabe, y con bloques cuadrados tocan aristas opuestas y generan un par que
  hace girar el objeto.
