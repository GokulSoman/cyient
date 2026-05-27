cmake_minimum_required(VERSION 3.8)
project(gait_intent)

# ── Find packages ─────────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# ── Install Python package ────────────────────────────────────────────────────
ament_python_install_package(${PROJECT_NAME})

# ── Install scripts ───────────────────────────────────────────────────────────
install(
  PROGRAMS
    scripts/run_pipeline.py
  DESTINATION lib/${PROJECT_NAME}
)

# ── Install config files ──────────────────────────────────────────────────────
install(
  DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# ── Install resource index marker ─────────────────────────────────────────────
install(
  FILES resource/${PROJECT_NAME}
  DESTINATION share/ament_index/resource_index/packages
)

# ── Tests ─────────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
