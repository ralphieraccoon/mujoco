// Copyright 2023 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MUJOCO_PLUGIN_ACTUATOR_TSA_H_
#define MUJOCO_PLUGIN_ACTUATOR_TSA_H_

#include <memory>
#include <optional>
#include <vector>

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtnum.h>

namespace mujoco::plugin::actuator {

struct TsaConfig {

  // Dynamics paramters
  double K_L = 0.0;
  double B = 0.0;
  double J = 0.0;
  double K_t = 0.0;

  // TSA parameters
  double l_u = 0.0;
  double r_s = 0.0;

  // Motor limits
  double v_limit = 0.0;
  double a_limit = 0.0;

  // Control mode 
  bool force_control_mode = true;

  // Reads plugin attributes to construct Tsa configuration.
  static TsaConfig FromModel(const mjModel* m, int instance);
};

// An actuator plugin which implements configurable PID control.
class Tsa {
 public:
  // Returns an instance of Tsa. The result can be null in case of
  // misconfiguration.
  static std::unique_ptr<Tsa> Create(const mjModel* m, int instance);

  // Returns the number of state variables for the plugin instance
  static int StateSize(const mjModel* m, int instance);

  // Resets the C++ Pid instance's state.
  // plugin_state is a C array pointer into mjData->plugin_state, with a size
  // equal to the value returned from StateSize.
  void Reset(mjtNum* plugin_state);

  // Computes the rate of change for activation variables
  void ActDot(const mjModel* m, mjData* d, int instance) const;

  // Idempotent computation which updates d->actuator_force and the internal
  // state of the class. Called after ActDot.
  void Compute(const mjModel* m, mjData* d, int instance);

  // Updates plugin state.
  void Advance(const mjModel* m, mjData* d, int instance) const;

  // Adds the TSA plugin to the global registry of MuJoCo plugins.
  static void RegisterPlugin();

 private:
  Tsa(TsaConfig config, std::vector<int> actuators);

  // Returns the expected number of activation variables for the instance.
  static int ActDim(const mjModel* m, int instance, int actuator_id);

  struct State {
    // Empty for now.
  };
  // Reads data from d->act and returns it as a State struct.
  State GetState(const mjModel* m, mjData* d, int actuator_idx) const;

  // Returns the Tsa force setpoint.
  mjtNum GetCtrl(const mjModel* m, const mjData* d, int actuator_idx,
                 const State& state, bool actearly) const;

  TsaConfig config_;
  // set of actuator IDs controlled by this plugin instance.
  std::vector<int> actuators_;
};

}  // namespace mujoco::plugin::actuator

#endif  // MUJOCO_PLUGIN_ACTUATOR_TSA_H_
