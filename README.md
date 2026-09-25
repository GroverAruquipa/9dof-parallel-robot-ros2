# Robot paralelo de 9 GDL (5P̲SS-S-4P̲SS) en ROS 2

Descripción, cinemática y visualización del robot paralelo de 9 GDL con
capacidad de agarre (Aruquipa, Lambert y Gosselin, Université Laval).

![URDF frente al CAD](docs/urdf_vs_cad.png)

| Paquete | Contenido |
|---|---|
| `ninedof_description` | Mallas STL, `config/geometry.yaml` (medido del CAD) y URDF/xacro |
| `ninedof_kinematics` | IK analítica, FK Gauss-Newton, matrices J y K, nodo `pose_to_joint_states` y tests |
| `ninedof_bringup` | Launch y configuración de RViz |

```bash
colcon build --symlink-install --packages-up-to ninedof_bringup
source install/setup.bash
ros2 launch ninedof_bringup view_robot.launch.py            # demo animada de los 9 GDL
ros2 launch ninedof_bringup view_robot.launch.py demo:=false
ros2 topic pub --once /pose_cmd std_msgs/msg/Float64MultiArray \
  "data: [0.0, 0.0, 0.1467, 0.1, 0.0, 0.0, 0.0, 0.0, 0.3]"   # [x y z a1 a2 a3 b1 b2 b3]
colcon test --packages-select ninedof_kinematics && colcon test-result --verbose
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
