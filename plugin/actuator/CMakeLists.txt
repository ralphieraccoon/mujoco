# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(MUJOCO_ACTUATOR_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../..
                            ${CMAKE_CURRENT_SOURCE_DIR}/../../src
)

set(MUJOCO_ACTUATOR_SRCS
    pid.cc
    pid.h
    tsa.cc
    tsa.h
    register.cc
)

add_compile_definitions(EIGEN_MPL2_ONLY)
findorfetch(
  USE_SYSTEM_PACKAGE
  OFF
  PACKAGE_NAME
  Eigen3
  LIBRARY_NAME
  eigen
  GIT_REPO
  https://gitlab.com/libeigen/eigen
  GIT_TAG
  75bcd155c40cb48e647c87c3f29052360255bc9e
  TARGETS
  Eigen3::Eigen
  EXCLUDE_FROM_ALL
)

add_library(actuator SHARED)
target_sources(actuator PRIVATE ${MUJOCO_ACTUATOR_SRCS})
target_include_directories(actuator PRIVATE ${MUJOCO_ACTUATOR_INCLUDE})
target_link_libraries(actuator PRIVATE mujoco Eigen3::Eigen)
target_compile_options(
  actuator
  PRIVATE ${AVX_COMPILE_OPTIONS}
          ${MUJOCO_MACOS_COMPILE_OPTIONS}
          ${EXTRA_COMPILE_OPTIONS}
          ${MUJOCO_CXX_FLAGS}
)
target_link_options(
  actuator
  PRIVATE
  ${MUJOCO_MACOS_LINK_OPTIONS}
  ${EXTRA_LINK_OPTIONS}
)
