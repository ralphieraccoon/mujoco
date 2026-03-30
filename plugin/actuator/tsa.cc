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

#include "tsa.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <utility>
#include <vector>
#include <Eigen/Core>

#include <mujoco/mujoco.h>

namespace mujoco::plugin::actuator {
namespace {

constexpr char kAttrK_L[] = "K_L";
constexpr char kAttrB[] = "B";
constexpr char kAttrJ[] = "J";
constexpr char kAttrK_t[] = "K_t";
constexpr char kAttrl_u[] = "l_u";
constexpr char kAttrr_s[] = "r_s";
constexpr char kAttrv_limit[] = "v_limit";
constexpr char kAttra_limit[] = "a_limit";
constexpr char kAttrforce_control_mode[] = "force_control_mode";

std::optional<mjtNum> ReadOptionalDoubleAttr(const mjModel* m, int instance,
                                             const char* attr) {
  const char* value = mj_getPluginConfig(m, instance, attr);
  if (value == nullptr || value[0] == '\0') {
    return std::nullopt;
  }
  return std::strtod(value, nullptr);
}

std::optional<mjtByte> ReadOptionalBoolAttr(const mjModel* m, int instance,
                                             const char* attr) {
  const char* value = mj_getPluginConfig(m, instance, attr);
  if (value == nullptr || value[0] == '\0') {
    return std::nullopt;
  }
  return (bool)std::stoi(value, nullptr);
}

mjtNum NextActivation(const mjModel* m, const mjData* d, int actuator_id,
                      int act_adr, mjtNum act_dot) {
  mjtNum act = d->act[act_adr];

  if (m->actuator_dyntype[actuator_id] == mjDYN_FILTEREXACT) {
    // exact filter integration
    // act_dot(0) = (ctrl-act(0)) / tau
    // act(h) = act(0) + (ctrl-act(0)) (1 - exp(-h / tau))
    //        = act(0) + act_dot(0) * tau * (1 - exp(-h / tau))
    mjtNum tau = mju_max(mjMINVAL, m->actuator_dynprm[actuator_id * mjNDYN]);
    act = act + act_dot * tau * (1 - mju_exp(-m->opt.timestep / tau));
  } else {
    // Euler integration
    act = act + act_dot * m->opt.timestep;
  }

  // clamp to actrange
  if (m->actuator_actlimited[actuator_id]) {
    mjtNum* actrange = m->actuator_actrange + 2 * actuator_id;
    act = mju_clip(act, actrange[0], actrange[1]);
  }

  return act;
}

mjtNum Tsa_h_func(mjtNum theta, TsaConfig config) {

  return theta*std::pow(config.r_s, 2) / std::sqrt(std::pow(config.l_u, 2) - std::pow(theta, 2)*std::pow(config.r_s, 2));
}

mjtNum Tsa_k_func(mjtNum theta, mjtNum lambda, TsaConfig config) {

  return lambda - std::sqrt(std::pow(config.l_u, 2) - std::pow(theta, 2)*std::pow(config.r_s, 2));
}

} // namespace

TsaConfig TsaConfig::FromModel(const mjModel* m, int instance) {
  TsaConfig config;
  config.K_L = ReadOptionalDoubleAttr(m, instance, kAttrK_L).value_or(0);
  config.B = ReadOptionalDoubleAttr(m, instance, kAttrB).value_or(0);
  config.J = ReadOptionalDoubleAttr(m, instance, kAttrJ).value_or(0);
  config.K_t = ReadOptionalDoubleAttr(m, instance, kAttrK_t).value_or(0);
  config.l_u = ReadOptionalDoubleAttr(m, instance, kAttrl_u).value_or(0);
  config.r_s = ReadOptionalDoubleAttr(m, instance, kAttrr_s).value_or(0);
  config.v_limit = ReadOptionalDoubleAttr(m, instance, kAttrv_limit).value_or(0);
  config.a_limit = ReadOptionalDoubleAttr(m, instance, kAttra_limit).value_or(0);

  config.force_control_mode = ReadOptionalBoolAttr(m, instance, kAttrforce_control_mode).value_or(false);

  return config;
}

std::unique_ptr<Tsa> Tsa::Create(const mjModel* m, int instance) {
  TsaConfig config = TsaConfig::FromModel(m, instance);

  if (config.K_L < 0) {
    mju_warning("coeffcient K_L must be positive.");
    return nullptr;
  }

  if (config.B < 0) {
    mju_warning("coeffcient B must be positive.");
    return nullptr;
  }

  if (config.J < 0) {
    mju_warning("coeffcient J must be positive.");
    return nullptr;
  }

  if (config.K_t < 0) {
    mju_warning("coeffcient K_t must be positive.");
    return nullptr;
  }

  if (config.l_u < 0) {
    mju_warning("coeffcient l_u must be positive.");
    return nullptr;
  }

  if (config.r_s < 0) {
    mju_warning("coeffcient r_s must be positive.");
    return nullptr;
  }

  if (config.v_limit < 0) {
    mju_warning("coeffcient v_limit must be positive.");
    return nullptr;
  }

  if (config.a_limit < 0) {
    mju_warning("coeffcient a_limit must be positive.");
    return nullptr;
  }

  std::vector<int> actuators;
  for (int i = 0; i < m->nu; i++) {
    if (m->actuator_plugin[i] == instance) {
      actuators.push_back(i);
    }
  }
  if (actuators.empty()) {
    mju_warning("actuator not found for plugin instance %d", instance);
    return nullptr;
  }
  // Validate actnum values for all actuators:
  for (int actuator_id : actuators) {
    int actnum = m->actuator_actnum[actuator_id];
    int expected_actnum = Tsa::ActDim(m, instance, actuator_id);
    int dyntype = m->actuator_dyntype[actuator_id];
    if (dyntype == mjDYN_FILTER || dyntype == mjDYN_FILTEREXACT ||
        dyntype == mjDYN_INTEGRATOR) {
      expected_actnum++;
    }
    if (actnum != expected_actnum) {
      mju_warning(
          "actuator %d has actdim %d, expected %d. Add actdim=\"%d\" to the "
          "actuator plugin element.",
          actuator_id, actnum, expected_actnum, expected_actnum);
      return nullptr;
    }
  }
  return std::unique_ptr<Tsa>(new Tsa(config, std::move(actuators)));
}

void Tsa::Reset(mjtNum* plugin_state) {}

mjtNum Tsa::GetCtrl(const mjModel* m, const mjData* d, int actuator_idx,
                    const State& state,
                    bool actearly) const {
  mjtNum ctrl = 0;
  if (m->actuator_dyntype[actuator_idx] == mjDYN_NONE) {
    ctrl = d->ctrl[actuator_idx];
  } else {
    // Use of act instead of ctrl, to create integrated-velocity controllers or
    // to filter the controls.
    int actadr = m->actuator_actadr[actuator_idx] +
                 m->actuator_actnum[actuator_idx] - 1;
    if (actearly) {
      ctrl = NextActivation(m, d, actuator_idx, actadr, d->act_dot[actadr]);
    } else {
      ctrl = d->act[actadr];
    }
  }
  return ctrl;
}

void Tsa::ActDot(const mjModel* m, mjData* d, int instance) const {
  for (int actuator_idx : actuators_) {
    State state = GetState(m, d, actuator_idx);
    mjtNum ctrl = GetCtrl(m, d, actuator_idx, state, /*actearly=*/false);

    int state_idx = m->actuator_actadr[actuator_idx];

    Eigen::Vector2d A(d->act_dot[state_idx + 1], -(config_.K_L/config_.J)*Tsa_h_func(d->act[state_idx], config_)*Tsa_k_func(d->act[state_idx], d->actuator_length[state_idx], config_)-(config_.B/config_.J)*d->act_dot[state_idx + 1]); 
    Eigen::Vector2d B(0, config_.K_t / config_.J);
    // if (config_.i_gain) {
    //   mjtNum integral = state.integral + error * m->opt.timestep;
    //   if (config_.i_max.has_value()) {
    //     integral = mju_clip(integral, -*config_.i_max, *config_.i_max);
    //   }
    //   d->act_dot[state_idx] = (integral - d->act[state_idx]) / m->opt.timestep;
    //   ++state_idx;
    // }
    // if (config_.slew_max.has_value()) {
    //   d->act_dot[state_idx] = (ctrl - d->act[state_idx]) / m->opt.timestep;
    //   ++state_idx;
    // }

    Eigen::Vector2d x_dot = A + B*ctrl;

    d->act_dot[state_idx] = x_dot[0];
    d->act_dot[state_idx + 1] = x_dot[1];

  }
}

void Tsa::Compute(const mjModel* m, mjData* d, int instance) {
  for (int i = 0; i < actuators_.size(); i++) {
    int actuator_idx = actuators_[i];
    int state_idx = m->actuator_actadr[actuator_idx];
    // State state = GetState(m, d, actuator_idx);
    // mjtNum ctrl =
    //     GetCtrl(m, d, actuator_idx, state, m->actuator_actearly[actuator_idx]);
    d->actuator_force[actuator_idx] = mju_clip(config_.K_L*Tsa_k_func(d->act[state_idx], *d->actuator_length, config_), 0, std::numeric_limits<mjtNum>::infinity());
  }
}

void Tsa::Advance(const mjModel* m, mjData* d, int instance) const {
  // act variables already updated by MuJoCo integrating act_dot
}

int Tsa::StateSize(const mjModel* m, int instance) {
  return 0;
}

int Tsa::ActDim(const mjModel* m, int instance, int actuator_id) {
  // double i_gain = ReadOptionalDoubleAttr(m, instance, kAttrIGain).value_or(0);
  // return (i_gain ? 1 : 0) + (HasSlew(m, instance) ? 1 : 0);
  return 2;
}

Tsa::State Tsa::GetState(const mjModel* m, mjData* d, int actuator_idx) const {
  State state;
  return state;
}

void Tsa::RegisterPlugin() {
  mjpPlugin plugin;
  mjp_defaultPlugin(&plugin);
  plugin.name = "mujoco.pid";
  plugin.capabilityflags |= mjPLUGIN_ACTUATOR;

  std::vector<const char*> attributes = {kAttrK_L, kAttrB, kAttrJ,
                                         kAttrK_t, kAttrl_u, kAttrr_s, kAttrv_limit, kAttra_limit};
  plugin.nattribute = attributes.size();
  plugin.attributes = attributes.data();
  plugin.nstate = Tsa::StateSize;

  plugin.init = +[](const mjModel* m, mjData* d, int instance) {
    std::unique_ptr<Tsa> tsa = Tsa::Create(m, instance);
    if (tsa == nullptr) {
      return -1;
    }
    d->plugin_data[instance] = reinterpret_cast<uintptr_t>(tsa.release());
    return 0;
  };
  plugin.destroy = +[](mjData* d, int instance) {
    delete reinterpret_cast<Tsa*>(d->plugin_data[instance]);
    d->plugin_data[instance] = 0;
  };
  plugin.reset = +[](const mjModel* m, mjtNum* plugin_state, void* plugin_data,
                     int instance) {
    auto* tsa = reinterpret_cast<Tsa*>(plugin_data);
    tsa->Reset(plugin_state);
  };
  plugin.actuator_act_dot = +[](const mjModel* m, mjData* d, int instance) {
    auto* tsa = reinterpret_cast<Tsa*>(d->plugin_data[instance]);
    tsa->ActDot(m, d, instance);
  };
  plugin.compute =
      +[](const mjModel* m, mjData* d, int instance, int capability_bit) {
        auto* tsa = reinterpret_cast<Tsa*>(d->plugin_data[instance]);
        tsa->Compute(m, d, instance);
      };
  plugin.advance = +[](const mjModel* m, mjData* d, int instance) {
    auto* tsa = reinterpret_cast<Tsa*>(d->plugin_data[instance]);
    tsa->Advance(m, d, instance);
  };
  // TODO: b/303823996 - allow actuator plugins to compute their derivatives wrt
  // qvel, for implicit integration
  mjp_registerPlugin(&plugin);
}

Tsa::Tsa(TsaConfig config, std::vector<int> actuators)
    : config_(std::move(config)), actuators_(std::move(actuators)) {}

}  // namespace mujoco::plugin::actuator
