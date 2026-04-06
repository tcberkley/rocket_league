use glam::{Mat4, Vec3};

pub struct Camera {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fovy: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    pub fn default_view() -> Self {
        Self {
            // Elevated above and behind the blue goal, looking toward centre field
            // Render space: Y=up, Z=−RL_Y (so blue goal is at Z≈+5120)
            eye: Vec3::new(0.0, 2500.0, 8000.0),
            target: Vec3::new(0.0, 200.0, 0.0),
            up: Vec3::Y,
            fovy: 70_f32.to_radians(),
            aspect: 1.0,
            near: 10.0,
            far: 30000.0,
        }
    }

    pub fn view_proj(&self) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye, self.target, self.up);
        let proj = Mat4::perspective_rh(self.fovy, self.aspect, self.near, self.far);
        proj * view
    }

    pub fn orbit(&mut self, dx: f32, dy: f32) {
        let radius = (self.eye - self.target).length();
        let dir = (self.eye - self.target).normalize();

        // Yaw around world Y
        let yaw = glam::Quat::from_rotation_y(dx * 0.005);
        let dir = yaw * dir;

        // Pitch around camera right
        let right = dir.cross(self.up).normalize();
        let pitch = glam::Quat::from_axis_angle(right, -dy * 0.005);
        let dir = pitch * dir;

        // Clamp vertical angle to prevent flipping
        let clamped = dir.normalize();
        if clamped.y.abs() < 0.98 {
            self.eye = self.target + clamped * radius;
        } else {
            self.eye = self.target + dir.normalize() * radius;
        }
    }

    pub fn zoom(&mut self, delta: f32) {
        let dir = self.eye - self.target;
        let new_len = (dir.length() * (1.0 - delta * 0.1)).max(500.0).min(20000.0);
        self.eye = self.target + dir.normalize() * new_len;
    }

    pub fn set_aspect(&mut self, width: u32, height: u32) {
        self.aspect = width as f32 / height.max(1) as f32;
    }

    /// Follow car: preserve the current eye→target offset so the user can
    /// still orbit freely while the camera tracks the car.
    pub fn follow_car(&mut self, car_pos: Vec3) {
        let offset = self.eye - self.target;
        self.target = car_pos;
        self.eye = car_pos + offset;
    }
}
