cmake_minimum_required(VERSION 3.8)
project(catheter_sim)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

# ── Generate custom message ────────────────────────────────────────
# NOTE: rosidl_generator_py will automatically install the generated
# Python bindings (catheter_sim/msg/*.py) via ament_python_install_package.
# Do NOT call ament_python_install_package again — it would conflict.
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/VesselDetection.msg"
  DEPENDENCIES std_msgs
)

# ── Install Python node scripts as executables ────────────────────
# This makes them accessible via `ros2 run catheter_sim <script_name>`
install(PROGRAMS
  catheter_sim/vessel_detector.py
  catheter_sim/image_publisher.py
  catheter_sim/target_planner.py
  catheter_sim/arm_controller.py
  DESTINATION lib/${PROJECT_NAME}
)

# ── Install the Python package __init__.py so imports work ────────
install(FILES
  catheter_sim/__init__.py
  DESTINATION lib/python3/dist-packages/${PROJECT_NAME}
)

# ── Install shared resource directories ───────────────────────────
install(DIRECTORY
  launch
  urdf
  config
  worlds
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
