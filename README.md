# Robot paralelo de 9 GDL (5P̲SS-S-4P̲SS) en ROS 2

Descripción, cinemática y visualización del robot paralelo de 9 GDL con
capacidad de agarre (Aruquipa, Lambert y Gosselin, Université Laval).

![URDF frente al CAD](docs/urdf_vs_cad.png)

| Paquete | Contenido |
|---|---|
| `ninedof_description` | Mallas STL, `config/geometry.yaml` (medido del CAD) y URDF/xacro |
| `ninedof_kinematics` | IK analítica, FK Gauss-Newton, matrices J y K, nodo `pose_to_joint_states` y tests |
| `ninedof_controllers` | Controlador ros2_control `CartesianPoseController` (C++): recibe la pose de las plataformas, interpola en espacio cartesiano y resuelve la IK en cada ciclo |
| `ninedof_bringup` | Launch, configuración de controladores y RViz |
| `ninedof_mujoco` | Video de capacidades, plugin de estado para `mujoco_ros2_control` y demo experimental de pick-and-place |

```bash
rosdep install --from-paths src --ignore-src -y
colcon build --symlink-install --packages-up-to ninedof_bringup
source install/setup.bash
```

**Con ros2_control** (hardware simulado `mock_components/GenericSystem`):

```bash
ros2 launch ninedof_bringup ninedof.launch.py              # demo de los 9 GDL
ros2 launch ninedof_bringup ninedof.launch.py demo:=false  # esperar comandos
ros2 topic pub --once /cartesian_pose_controller/pose_cmd std_msgs/msg/Float64MultiArray \
  "data: [0.0, 0.0, 0.1467, 0.1, 0.0, 0.0, 0.0, 0.0, 0.3]"   # [x y z a1 a2 a3 b1 b2 b3]
ros2 topic echo /platform_pose                             # pose medida por la FK
ros2 control list_controllers
```

```
pose_cmd ─► cartesian_pose_controller ─► 9 actuadores (position) ─► hardware
                (interpolación + IK)                                    │
RViz ◄─ robot_state_publisher ◄─ /joint_states ◄─ joint_state_broadcaster
                                     ▲
                        fk_joint_state_publisher (FK: articulaciones pasivas + /platform_pose)
```

El controlador interpola la **pose** (no los actuadores): así cada comando es
una solución de la IK y respeta los lazos cerrados. Las poses fuera del
espacio de trabajo o de la carrera (±25 mm) se rechazan.

**Con física en MuJoCo** (cadenas cerradas con restricciones `connect`, plugin
[`mujoco_ros2_control`](https://github.com/ros-controls/mujoco_ros2_control)):

```bash
ros2 launch ninedof_bringup ninedof.launch.py sim:=mujoco            # visor de MuJoCo + RViz
ros2 launch ninedof_bringup ninedof.launch.py sim:=mujoco headless:=true
```

El modelo MJCF se genera desde `geometry.yaml` y `dynamics.yaml` (masas
**estimadas**, a reemplazar por las de SolidWorks):

```bash
cd src/ninedof_description && python3 scripts/generate_mjcf.py
```

⚠️ Con la geometría actual del CAD la plataforma 2 tiene una torsión casi
libre y en simulación cambia de modo de ensamblaje: ver
[docs/nota_diseno_torsion.md](docs/nota_diseno_torsion.md).

### Capacidades del robot (video)

![Grados de libertad](docs/showcase_frames.png)

[docs/showcase.mp4](docs/showcase.mp4) (50 s, 1080p): traslaciones X, Y, Z;
rotaciones de las dos plataformas juntas; rotación relativa entre plataformas
(la pinza); una trayectoria circular y un cono. Es una animación cinemática:
cada cuadro es la pose exacta de la trayectoria, con los actuadores dados por
la IK. Las patas se colorean según la plataforma que mueven (rojo: 5, azul: 4).

```bash
ros2 run ninedof_mujoco showcase_video --check                 # verifica la trayectoria
MUJOCO_GL=osmesa ros2 run ninedof_mujoco showcase_video -o showcase.mp4
ros2 launch ninedof_bringup view_robot.launch.py               # la misma secuencia en RViz
```

| Movimiento | Amplitud | Límite al 80 % de la carrera |
|---|---|---|
| Traslación X / Y / Z | ±30 / ±30 / ±15 mm | ±53 / ±48 / ±19 mm |
| Rotación roll / pitch / yaw | ±20° / ±20° / ±45° | ±33° / ±30° / ±80° |
| Rotación relativa (pinza) | 0–30° por plataforma | 33° |

La trayectoria usa como máximo 18 de los 25 mm de carrera de los actuadores.

> `ninedof_mujoco` también incluye una demo experimental de pick-and-place
> (`pick_place.launch.py`) con agarre virtual; los dedos del CAD son pequeños
> para agarrar por fricción (ver [nota de diseño](docs/nota_diseno_torsion.md)).

**Solo visualización** (sin ros2_control):

```bash
ros2 launch ninedof_bringup view_robot.launch.py
```

**Tests** (IK/FK contra el CAD, Jacobianos, controlador):

```bash
colcon test --packages-select ninedof_kinematics ninedof_controllers
colcon test-result --verbose
```

URDF no admite cadenas cerradas, así que el robot se describe como un árbol
(actuadores + barras distales, y una cadena virtual de 6 GDL hasta la
plataforma 1 más la esférica central hasta la plataforma 2). El nodo
`pose_to_joint_states` cierra los lazos numéricamente con la cinemática inversa.

# ROS 2 en la nube (GitHub Codespaces)

Este repositorio trae un entorno de ROS 2 **Jazzy** ya configurado, con
`ros2_control`, `ros2_controllers` y Gazebo. No necesitas instalar nada en tu
computadora: todo corre en un servidor de GitHub y lo usas desde el navegador.

## Cómo abrirlo

1. En la página del repositorio en GitHub, pulsa el botón verde **Code**.
2. Abre la pestaña **Codespaces** y pulsa **Create codespace on main**.
3. Espera a que se construya (la primera vez tarda ~5–10 minutos; después es rápido).
4. Se abre VS Code en el navegador, con una terminal en la que ROS 2 ya está cargado.

Pruébalo:

```bash
ros2 run demo_nodes_cpp talker
# en otra terminal:
ros2 run demo_nodes_cpp listener
```

## Ver interfaces gráficas (RViz, Gazebo, rqt)

1. Abre la pestaña **Ports** (junto a la terminal).
   Si no la ves: `Ctrl + Shift + P` → **Ports: Focus on Ports View**.
2. Busca el puerto **6080** ("Escritorio (noVNC)") y pulsa el icono del globo 🌐.
   Si no aparece, pulsa **Add Port** y escribe `6080`.
3. **Agrega `/vnc.html` al final de la dirección** que se abre
   (ej. `https://...-6080.app.github.dev/vnc.html`), pulsa **Connect** y usa la contraseña `ros`.
4. Todo lo que abras desde la terminal (por ejemplo `rviz2` o `gz sim`) aparece ahí.

## Compilar tus paquetes

Pon tus paquetes dentro de la carpeta `src/` y compila desde la raíz:

```bash
sudo apt-get update && rosdep install --from-paths src --ignore-src -y
colcon build --symlink-install
source install/setup.bash
```

Si quieres el código fuente de ros2_control para estudiarlo o modificarlo:

```bash
git clone -b jazzy https://github.com/ros-controls/ros2_control.git src/ros2_control
git clone -b jazzy https://github.com/ros-controls/ros2_control_demos.git src/ros2_control_demos
sudo apt-get update && rosdep install --from-paths src --ignore-src -y
colcon build --symlink-install
```

## Notas sobre el uso gratuito

- Las cuentas personales de GitHub tienen horas gratis de Codespaces cada mes.
  Una máquina de 4 núcleos gasta esas horas el doble de rápido que una de 2.
- **Detén el codespace cuando no lo uses** (github.com/codespaces → `...` → *Stop codespace*).
  Se detiene solo tras 30 minutos sin actividad, y tus archivos se conservan.
- Haz `git commit` y `git push` de tu trabajo con frecuencia.
