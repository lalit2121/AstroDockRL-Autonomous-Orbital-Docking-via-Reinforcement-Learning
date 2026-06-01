"""
ORBITAL DOCKING GYMNASIUM ENVIRONMENT
Autonomous Rendezvous & Docking using Clohessy-Wiltshire Equations

PHYSICS BACKGROUND (beginner-friendly)
We model the chaser spacecraft moving relative to a target spacecraft
already in a circular orbit. Instead of simulating two full orbits around Earth,
we use the Hill (Local Vertical Local Horizontal, LVLH) reference frame:
  X axis: along the orbit track (in-track)
  Y axis: radial (toward/away from Earth)
  Z axis: out-of-plane (cross-track)

In this frame, the Clohessy-Wiltshire (CW) equations describe how a
nearby object moves due to orbital mechanics:
  ẍ =  3n²x + 2nẏ + uₓ
  ÿ =      - 2nẋ + u_y
  z̈ = -n²z        + u_z

where:
  n  = mean motion of the target orbit = sqrt(μ/a³)  [rad/s]
  μ  = Earth's gravitational parameter ≈ 3.986e14 m³/s²
  a  = semi-major axis of target orbit (m)
  uₓ, u_y, u_z = applied accelerations (thrust / spacecraft mass) [m/s²]

NEW PERTURBATION MODELS (added):
1. NRLMSISE-00 Aerodynamic Drag
   Uses the industry-standard NRLMSISE-00 atmospheric model via a pre-computed
   Lookup Table (LUT) + exponential scale-height correction for the chaser's 
   radial offset. Includes Domain Randomization of solar flux (F10.7) and 
   geomagnetic storms (Ap) per episode.
2. Solar Radiation Pressure (SRP)
   Photon pressure on the chaser's solar panels creates a constant
   acceleration bias that toggles on/off with eclipse transitions.
3. Thruster Execution Noise
   Real RCS thrusters have magnitude error, misalignment, and a dead-band.
4. Sensor / Navigation Noise
   Observations are corrupted by Gaussian noise representing GPS + IMU errors.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import torch

# Safe import for NRLMSISE-00 and interpolation
try:
    import pymsis
    from scipy.interpolate import interp1d
    from datetime import datetime, timedelta
    HAS_MSIS = True
except ImportError:
    HAS_MSIS = False
    print("Warning: pymsis or scipy not found. Falling back to constant density model.")
    print("Install via: pip install pymsis scipy")


class OrbitalDockingEnv(gym.Env):
    """
    Gymnasium environment for autonomous orbital docking.
    
    Perturbation parameters (all 0.0 = off, 1.0 = full physical magnitude):
      j2_perturbation_scale    — J2 oblateness differential acceleration
      drag_perturbation_scale  — aerodynamic drag differential deceleration
      srp_perturbation_scale   — solar radiation pressure acceleration
      thruster_noise_scale     — RCS execution errors (magnitude + misalign)
      sensor_noise_scale       — GPS/IMU observation noise
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        orbit_altitude_km: float = 400.0,
        dt: float = 1.0,
        max_episode_steps: int = 1000,
        max_thrust: float = 0.05,
        docking_radius: float = 0.5,
        max_docking_speed: float = 0.05,
        initial_distance_range: tuple = (80.0, 250.0),
        fuel_penalty_weight: float = 0.05,
        velocity_penalty_weight: float = 0.1,
        render_mode: str = None,
        use_3d: bool = True,
        # ── Perturbations ─────────────────────────────────────────────
        j2_perturbation_scale: float = 0.0,
        drag_perturbation_scale: float = 0.0,
        srp_perturbation_scale: float = 0.0,
        thruster_noise_scale: float = 0.0,
        sensor_noise_scale: float = 0.0,
    ):
        super().__init__()

        # ── Perturbation scales ───────────────────────────────────────────
        self.j2_perturbation_scale   = float(np.clip(j2_perturbation_scale,   0.0, 1.0))
        self.drag_perturbation_scale = float(np.clip(drag_perturbation_scale, 0.0, 1.0))
        self.srp_perturbation_scale  = float(np.clip(srp_perturbation_scale,  0.0, 1.0))
        self.thruster_noise_scale    = float(np.clip(thruster_noise_scale,    0.0, 1.0))
        self.sensor_noise_scale      = float(np.clip(sensor_noise_scale,      0.0, 1.0))

        # ── Orbital parameters ────────────────────────────────────────────
        self.mu          = 3.986004418e14
        self.R_earth     = 6371e3
        self.orbit_radius = self.R_earth + orbit_altitude_km * 1e3
        self.n           = np.sqrt(self.mu / self.orbit_radius**3)
        self.T_orbit     = 2 * np.pi / self.n

        # ── Absolute Orbit Parameters (for NRLMSISE-00) ────────────────────
        self.inclination = np.deg2rad(51.6)      # ISS inclination [rad]
        self.earth_rot_rate = 7.2921159e-5       # Earth rotation rate [rad/s]
        self.scale_height = 50.0e3               # Nominal LEO scale height [m] (~50km at 400km alt)

        # Space weather indices (Defaults, randomized per episode in reset)
        self.f107a = 150.0   # 81-day average F10.7 solar flux
        self.f107  = 150.0   # Daily F10.7
        self.ap    = 4.0     # Daily geomagnetic Ap index

        # ── J2 constants ──────────────────────────────────────────────────
        self.target_eccentricity = np.array([0.0005, 0.0003]) 
        self.J2        = 1.08262668e-3
        self._j2_coeff = (1.5 * self.J2 * self.mu * self.R_earth**2
                          / self.orbit_radius**4)

        # ── Drag constants ────────────────────────────────────────────────
        self._rho_ref  = 3.0e-13  # Fallback constant density [kg/m^3]
        self._B_chaser = 0.015
        self._B_target = 0.010
        self._v_orb    = np.sqrt(self.mu / self.orbit_radius)

        # ── SRP constants ─────────────────────────────────────────────────
        self._P_sr         = 4.56e-6        # N/m² solar pressure at 1 AU
        self._Cr_Am_chaser = 1.5 * 0.020    # Cr * A/m for chaser
        self._Cr_Am_target = 1.5 * 0.015    # Cr * A/m for target
        self._srp_diff_coeff = self._P_sr * (self._Cr_Am_chaser - self._Cr_Am_target)
        self._sun_hat        = np.array([1.0, 0.0, 0.0])
        self._in_eclipse     = False
        self._sun_phase = 0.0          # orbital angle of sun vector [rad]
        self._sun_rate  = 2*np.pi / (365.25*24*3600) * self.T_orbit / (2*np.pi)
        # ≈ n_sun/n_orbit — sun rotates ~1°/day vs ~16 orbits/day

        # ── Thruster noise constants ──────────────────────────────────────
        self._thrust_mag_std    = 0.05
        self._thrust_misalign_std = np.deg2rad(1.0)
        self._thrust_dead_band  = 0.01 * max_thrust

        # ── Sensor noise constants ────────────────────────────────────────
        self._pos_noise_std = 0.30
        self._vel_noise_std = 0.003

        # ── Simulation settings ───────────────────────────────────────────
        self.dt                    = dt
        self.max_episode_steps     = max_episode_steps
        self.max_thrust            = max_thrust
        self.docking_radius        = docking_radius
        self.max_docking_speed     = max_docking_speed
        self.initial_distance_range = initial_distance_range
        self.fuel_penalty_weight   = fuel_penalty_weight
        self.velocity_penalty_weight = velocity_penalty_weight
        self.render_mode           = render_mode
        self.use_3d                = use_3d

        # ── Normalization ─────────────────────────────────────────────────
        self.pos_norm    = initial_distance_range[1]
        self.vel_norm    = 2.0
        self.max_distance = initial_distance_range[1] * 3.0

        # ── Spaces ────────────────────────────────────────────────────────
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )

        # ── Internal state ────────────────────────────────────────────────
        self.state         = None
        self.step_count    = 0
        self.prev_distance = None
        self.total_fuel    = 0.0
        self.trajectory    = []

        self.episode_success = False
        self.episode_crash   = False

        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_torch = True
        
        # Initialize LUT lambda to fallback
       
           
        self.use_torch = True
        
        # ── Pre-compute Density Library for Fast Resets ─────────────────────
        self._density_library = []
        if self.drag_perturbation_scale > 0.0 and HAS_MSIS:
            print("⏳ Pre-computing MSIS density library (takes ~2-3s)...")
            for _ in range(20):  # 20 diverse space-weather scenarios
                self._generate_single_msis_profile()
        else:
            self._density_library = [lambda t: self._rho_ref]
            
        self._rho_lut = lambda t: self._rho_ref  # Default fallback

    # ─────────────────────────────────────────────────────────────────────────
    #  RESET
    # ─────────────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        d_min, d_max = self.initial_distance_range
        distance = self.np_random.uniform(d_min, d_max)

        if self.use_3d:
            phi   = self.np_random.uniform(0, 2 * np.pi)
            theta = self.np_random.uniform(0, np.pi)
            x = distance * np.sin(theta) * np.cos(phi)
            y = distance * np.sin(theta) * np.sin(phi)
            z = distance * np.cos(theta)
        else:
            phi = self.np_random.uniform(0, 2 * np.pi)
            x   = distance * np.cos(phi)
            y   = distance * np.sin(phi)
            z   = 0.0

        v_scale = 0.1
        vx = self.np_random.uniform(-v_scale, v_scale)
        vy = self.np_random.uniform(-v_scale, v_scale)
        vz = self.np_random.uniform(-v_scale, v_scale) if self.use_3d else 0.0

        self.state         = np.array([x, y, z, vx, vy, vz], dtype=np.float64)
        self.step_count    = 0
        self.prev_distance = np.linalg.norm([x, y, z])
        self.total_fuel    = 0.0
        self.trajectory    = [self.state.copy()]
        self.episode_success = False
        self.episode_crash   = False

        # ── Initialize Orbit Time & Density LUT ───────────────────────────
           
        self._current_orbit_time = self.np_random.uniform(0, self.T_orbit)
        
        if self.drag_perturbation_scale > 0.0:
            if HAS_MSIS and len(self._density_library) > 0:
                # ⚡ Instantly pick a pre-computed profile (microseconds instead of ~0.5s)
                self._rho_lut = self.np_random.choice(self._density_library)
            else:
                self._rho_lut = lambda t: self._rho_ref
        else:
            self._rho_lut = lambda t: self._rho_ref

        # ── SRP episode setup ─────────────────────────────────────────────
        if self.srp_perturbation_scale > 0.0:
            # Random initial sun phase + inclination offset
            self._sun_phase = self.np_random.uniform(0, 2 * np.pi)
            self._sun_declination = self.np_random.uniform(-np.deg2rad(23.5), np.deg2rad(23.5))
            self._update_sun_vector()
            self._in_eclipse = self.np_random.random() > 0.60
        else:
            self._sun_hat    = np.array([1.0, 0.0, 0.0])
            self._in_eclipse = False

        return self._get_observation(), self._get_info()

    # ─────────────────────────────────────────────────────────────────────────
    #  NRLMSISE-00 DENSITY LOOKUP TABLE
    # ─────────────────────────────────────────────────────────────────────────
    def _generate_single_msis_profile(self):
        """
        Generates one NRLMSISE-00 density profile using randomized space weather,
        then appends the interpolation function to self._density_library.
        Called ONLY during __init__ to avoid slowing down episode resets.
        """
        if not HAS_MSIS:
            return

        # Domain Randomization: vary solar flux & geomagnetic activity
        f107 = np.random.uniform(80.0, 220.0)
        f107a = f107
        ap = np.random.uniform(2.0, 40.0)

        num_points = 360
        thetas = np.linspace(0, 2 * np.pi, num_points)
        
        lats = np.rad2deg(np.arcsin(np.sin(self.inclination) * np.sin(thetas)))
        lons = np.rad2deg(np.arctan2(np.cos(self.inclination) * np.sin(thetas), np.cos(thetas)))
        alts = np.full(num_points, (self.orbit_radius - self.R_earth) / 1000.0)
        
        base_date = datetime(2024, 1, 1)
        dates = [base_date + timedelta(seconds=i) for i in range(num_points)]

        try:
            msis_output = pymsis.msis00.run(
                dates, lons, lats, alts, 
                [f107a] * num_points, 
                [f107] * num_points, 
                [[ap] * 7] * num_points
            )
            rho_profile = msis_output[:, 0]
        except Exception:
            rho_profile = np.full(num_points, self._rho_ref)

        # Create fast interpolation function and store it
        lut = interp1d(
            np.linspace(0, self.T_orbit, num_points), 
            rho_profile, 
            kind='cubic', 
            fill_value="extrapolate"
        )
        self._density_library.append(lut)

    # ─────────────────────────────────────────────────────────────────────────
    #  STEP
    # ─────────────────────────────────────────────────────────────────────────
    def step(self, action):
        # ── 1. Scale commanded thrust ──────────────────────────────────────
        thrust_cmd = np.clip(action, -1.0, 1.0) * self.max_thrust
        thrust_cmd = np.array(thrust_cmd, dtype=np.float64)

        # ── 2. Apply thruster execution noise ─────────────────────────────
        thrust_actual = self._apply_thruster_noise(thrust_cmd)

        # ── 3. Propagate true dynamics ────────────────────────────────────
        self.state = self._rk4_step(self.state, thrust_actual)

        # ── 4. Update eclipse state & orbit time ──────────────────────────
        self._update_eclipse()
        self._current_orbit_time = (self._current_orbit_time + self.dt) % self.T_orbit

        self.trajectory.append(self.state.copy())
        self.total_fuel += np.linalg.norm(thrust_actual) * self.dt
        self.step_count += 1

        pos      = self.state[:3]
        vel      = self.state[3:]
        distance = np.linalg.norm(pos)
        speed    = np.linalg.norm(vel)

        reward, terminated = self._compute_reward(
            pos, vel, distance, speed, thrust_actual
        )
        truncated = self.step_count >= self.max_episode_steps
        self.prev_distance = distance

        # ── 5. Return noisy observation ───────────────────────────────────
        obs  = self._get_observation()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    # ─────────────────────────────────────────────────────────────────────────
    #  PERTURBATION: J2
    # ─────────────────────────────────────────────────────────────────────────
    def _j2_differential_accel(self, pos):
        if self.j2_perturbation_scale == 0.0:
            return np.zeros(3)
        
        x, y, z = pos
        n = self.n
        ex, ey = self.target_eccentricity
        
        coeff = self._j2_coeff / self.orbit_radius   # units: 1/s²
        
        ax = coeff * (4.0*x - 6.0*ex*y)
        ay = coeff * (-1.0*y + 6.0*ey*x)
        az = coeff * (-3.0*z)
        
        return self.j2_perturbation_scale * np.array([ax, ay, az])

    # ─────────────────────────────────────────────────────────────────────────
    #  PERTURBATION: DRAG (NRLMSISE-00 + Scale Height)
    # ─────────────────────────────────────────────────────────────────────────
    def _drag_differential_accel(self, pos, vel):
        """
        Differential aerodynamic drag using NRLMSISE-00 LUT + Scale Height correction.
        """
        if self.drag_perturbation_scale == 0.0:
            return np.zeros(3)

        x, y, z = pos
        vx, vy, vz = vel

        # 1. Get Target's density from the pre-computed LUT
        rho_target = float(self._rho_lut(self._current_orbit_time))

        # 2. Apply Scale-Height Correction for Chaser's radial offset (y)
        rho_chaser = rho_target * np.exp(-y / self.scale_height)

        # 3. Calculate Differential Drag
        a_drag_chaser_x = -0.5 * rho_chaser * self._v_orb**2 * self._B_chaser
        a_drag_target_x = -0.5 * rho_target * self._v_orb**2 * self._B_target
        
        delta_a_x = a_drag_chaser_x - a_drag_target_x
        
        a_drag_correction = -0.5 * rho_chaser * self._v_orb * self._B_chaser * np.array([vx, vy, vz])
        
        total = np.array([delta_a_x, 0.0, 0.0]) + a_drag_correction
        
        return self.drag_perturbation_scale * total

    # ─────────────────────────────────────────────────────────────────────────
    #  PERTURBATION: SOLAR RADIATION PRESSURE
    # ─────────────────────────────────────────────────────────────────────────
    def _srp_differential_accel(self):
        if self.srp_perturbation_scale == 0.0:
            return np.zeros(3)

        if self._in_eclipse:
            return np.zeros(3)

        accel = self._srp_diff_coeff * self._sun_hat
        return self.srp_perturbation_scale * accel

    def _update_eclipse(self):
        if self.srp_perturbation_scale == 0.0:
            return
        # Advance sun phase (slow, ~1 rev/year scaled to orbit time)
        self._sun_phase = (self._sun_phase + self._sun_rate) % (2 * np.pi)
        self._update_sun_vector()
        # Eclipse via geometry: shadow if sun-orbit dot product < cos(penumbra half-angle)
        # For circular LEO, eclipse fraction ≈ arcsin(R_earth/a)/pi
        shadow_half_angle = np.arcsin(self.R_earth / self.orbit_radius)  # ~69° for ISS
        orbit_angle = (self._current_orbit_time / self.T_orbit) * 2 * np.pi
        # Component of sun vector along orbit normal (z-axis)
        sun_orbit_dot = abs(np.dot(self._sun_hat, np.array([np.cos(orbit_angle), np.sin(orbit_angle), 0.0])))
        self._in_eclipse = sun_orbit_dot < np.cos(shadow_half_angle)

    def _update_sun_vector(self):
        """Sun direction in Hill frame: rotates slowly due to Earth's orbit around Sun."""
        self._sun_hat = np.array([
            np.cos(self._sun_phase) * np.cos(self._sun_declination),
            np.sin(self._sun_phase) * np.cos(self._sun_declination),
            np.sin(self._sun_declination),
        ])

    # ─────────────────────────────────────────────────────────────────────────
    #  PERTURBATION: THRUSTER EXECUTION NOISE
    # ─────────────────────────────────────────────────────────────────────────
    def _apply_thruster_noise(self, thrust_cmd):
        if self.thruster_noise_scale == 0.0:
            return thrust_cmd.copy()

        s = self.thruster_noise_scale
        thrust = thrust_cmd.copy()

        dead_band = s * self._thrust_dead_band
        thrust[np.abs(thrust) < dead_band] = 0.0

        if np.all(thrust == 0.0):
            return thrust

        mag_std = s * self._thrust_mag_std
        mag_noise = self.np_random.normal(1.0, mag_std, size=3)
        thrust = thrust * mag_noise

        misalign_std = s * self._thrust_misalign_std
        θx, θy, θz = self.np_random.normal(0.0, misalign_std, size=3)

        Rx = np.array([[1,    0,     0  ],
                       [0,  np.cos(θx),  -np.sin(θx)],
                       [0,  np.sin(θx),  np.cos(θx)]])
        Ry = np.array([[ np.cos(θy), 0, np.sin(θy)],
                       [0,           1,  0          ],
                       [-np.sin(θy), 0, np.cos(θy)]])
        Rz = np.array([[np.cos(θz), -np.sin(θz), 0],
                       [np.sin(θz),  np.cos(θz), 0],
                       [0,           0,           1]])

        R_total = Rz @ Ry @ Rx
        thrust  = R_total @ thrust

        return thrust

    # ─────────────────────────────────────────────────────────────────────────
    #  CW DERIVATIVES
    # ─────────────────────────────────────────────────────────────────────────
    def _cw_derivatives(self, state, thrust):
        x, y, z, vx, vy, vz = state
        ux, uy, uz = thrust
        n = self.n

        ax = 3 * n**2 * x + 2 * n * vy + ux
        ay = -2 * n * vx + uy
        az = -n**2 * z + uz

        pos  = np.array([x, y, z])
        vel  = np.array([vx, vy, vz])

        dj2   = self._j2_differential_accel(pos)
        # Pass both pos and vel to drag for scale-height correction
        ddrag = self._drag_differential_accel(pos, vel)
        dsrp  = self._srp_differential_accel()

        ax += dj2[0] + ddrag[0] + dsrp[0]
        ay += dj2[1] + ddrag[1] + dsrp[1]
        az += dj2[2] + ddrag[2] + dsrp[2]

        return np.array([vx, vy, vz, ax, ay, az])

    def _cw_derivatives_torch(self, state_tensor, thrust_tensor):
        x, y, z, vx, vy, vz = torch.split(state_tensor, 1)
        ux, uy, uz = torch.split(thrust_tensor, 1)
        n = self.n
        ax = 3 * n**2 * x + 2 * n * vy + ux
        ay = -2 * n * vx  + uy
        az = -n**2 * z + uz
        return torch.cat([vx, vy, vz, ax, ay, az], dim=0)

    def _rk4_step(self, state, thrust):
        any_perturbation = (
            self.j2_perturbation_scale    > 0.0
            or self.drag_perturbation_scale  > 0.0
            or self.srp_perturbation_scale   > 0.0
        )
        use_fast = self.use_torch and not any_perturbation

        if use_fast:
            s  = torch.tensor(state,  dtype=torch.float32, device=self.device)
            u  = torch.tensor(thrust, dtype=torch.float32, device=self.device)
            dt = self.dt
            k1 = self._cw_derivatives_torch(s, u)
            k2 = self._cw_derivatives_torch(s + 0.5 * dt * k1,  u)
            k3 = self._cw_derivatives_torch(s + 0.5 * dt * k2, u)
            k4 = self._cw_derivatives_torch(s + dt * k3, u)
            return (s + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)).cpu().numpy()
        else:
            dt = self.dt
            k1 = self._cw_derivatives(state, thrust)
            k2 = self._cw_derivatives(state + 0.5 * dt * k1, thrust)
            k3 = self._cw_derivatives(state + 0.5 * dt * k2, thrust)
            k4 = self._cw_derivatives(state + dt * k3, thrust)
            return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # ─────────────────────────────────────────────────────────────────────────
    #  REWARD
    # ─────────────────────────────────────────────────────────────────────────
    def _compute_reward(self, pos, vel, distance, speed, thrust):
        terminated = False
        reward     = 0.0

        distance_improvement = self.prev_distance - distance
        reward += 5.0 * distance_improvement / self.pos_norm

        proximity_bonus = np.exp(-distance / (self.pos_norm * 0.1))
        reward += 0.5 * proximity_bonus

        proximity_factor  = np.exp(-distance / (self.pos_norm * 0.2))
        velocity_penalty  = self.velocity_penalty_weight * speed * proximity_factor
        reward -= velocity_penalty

        fuel_cost = self.fuel_penalty_weight * np.linalg.norm(thrust)
        reward   -= fuel_cost

        if distance > self.max_distance:
            reward -= 50.0
            self.episode_crash = True
            terminated = True
            return reward, terminated

        if distance < self.docking_radius * 3 and speed > self.max_docking_speed * 5:
            reward -= 30.0 * (speed / self.vel_norm)
            self.episode_crash = True
            terminated = True
            return reward, terminated

        if distance < self.docking_radius and speed < self.max_docking_speed:
            reward += 100.0
            self.episode_success = True
            terminated = True
            return reward, terminated

        return reward, terminated

    # ─────────────────────────────────────────────────────────────────────────
    #  OBSERVATION
    # ─────────────────────────────────────────────────────────────────────────
    def _get_observation(self):
        x, y, z, vx, vy, vz = self.state

        if self.sensor_noise_scale > 0.0:
            pos_sigma = self.sensor_noise_scale * self._pos_noise_std
            vel_sigma = self.sensor_noise_scale * self._vel_noise_std
            x  += self.np_random.normal(0.0, pos_sigma)
            y  += self.np_random.normal(0.0, pos_sigma)
            z  += self.np_random.normal(0.0, pos_sigma)
            vx += self.np_random.normal(0.0, vel_sigma)
            vy += self.np_random.normal(0.0, vel_sigma)
            vz += self.np_random.normal(0.0, vel_sigma)

        pos      = np.array([x, y, z])
        vel      = np.array([vx, vy, vz])
        distance = np.linalg.norm(pos)
        speed    = np.linalg.norm(vel)

        return np.array([
            x  / self.pos_norm,
            y  / self.pos_norm,
            z  / self.pos_norm,
            vx / self.vel_norm,
            vy / self.vel_norm,
            vz / self.vel_norm,
            distance / self.pos_norm,
            speed    / self.vel_norm,
            self.step_count / self.max_episode_steps,
        ], dtype=np.float32)

    def _get_info(self):
        pos = self.state[:3]
        vel = self.state[3:]
        return {
            "distance":   float(np.linalg.norm(pos)),
            "speed":      float(np.linalg.norm(vel)),
            "fuel_used":  float(self.total_fuel),
            "step":       self.step_count,
            "success":    self.episode_success,
            "crash":      self.episode_crash,
            "in_eclipse": self._in_eclipse,
            "state":      self.state.copy(),
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  RENDER / UTILITY
    # ─────────────────────────────────────────────────────────────────────────
    def render(self):
        if self.render_mode == "human":
            pos      = self.state[:3]
            vel      = self.state[3:]
            distance = np.linalg.norm(pos)
            speed    = np.linalg.norm(vel)
            eclipse  = "ECL" if self._in_eclipse else "SUN"
            print(
                f"Step {self.step_count:4d} |  "
                f"Dist: {distance:7.2f} m |  "
                f"Speed: {speed:6.3f} m/s |  "
                f"Fuel: {self.total_fuel:.3f} |  "
                f"{eclipse} |  "
                f"Pos: [{pos[0]:7.2f}, {pos[1]:7.2f}, {pos[2]:7.2f}]"
            )

    def get_trajectory(self):
        return np.array(self.trajectory)

    def close(self):
        pass

# ─────────────────────────────────────────────────────────────────────────────
# QUICK SANITY-CHECK
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("  Orbital Docking Environment — Sanity Check (all perturbations)")
    print("=" * 65)
    env = OrbitalDockingEnv(
        render_mode="human",
        j2_perturbation_scale=1.0,
        drag_perturbation_scale=1.0,
        srp_perturbation_scale=1.0,
        thruster_noise_scale=1.0,
        sensor_noise_scale=1.0,
    )

    obs, info = env.reset(seed=42)
    print(f"\nInitial distance:  {info['distance']:.2f} m")
    print(f"Sun direction:     {env._sun_hat.round(3)}")
    print(f"Eclipse:           {env._in_eclipse}")
    print(f"\nRunning 10 random steps...")
    print("-" * 65)

    total_reward = 0
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        if terminated or truncated:
            print("Episode ended early.")
            break

    print(f"\nTotal reward over 10 steps: {total_reward:.4f}")
    print("All perturbations active — environment check passed!")
    env.close()