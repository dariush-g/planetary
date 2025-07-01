use glam::{Mat4, Vec2, Vec3};

#[derive(Debug, Default, Clone, Copy)]
pub struct Camera {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub aspect: f32,
    pub fovy: f32,
    pub znear: f32,
    pub zfar: f32,
}

impl Camera {
    pub fn build_view_projection_matrix(&self) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye, self.target, self.up);
        let proj = Mat4::perspective_rh(self.fovy, self.aspect, self.znear, self.zfar);
        proj * view
    }
}
#[derive(Clone, Debug, Default)]
pub struct OrbitCamera {
    pub theta: f32,  // left-right angle
    pub phi: f32,    // up-down angle
    pub radius: f32, // distance from target
    pub target: Vec3,
    pub sensitivity: f32,
    pub zoom_speed: f32,
}

impl OrbitCamera {
    pub fn rotate(&mut self, delta: glam::Vec2) {
        self.theta -= delta.x * self.sensitivity;
        self.phi -= delta.y * self.sensitivity;

        // Clamp phi to avoid flipping upside down
        let epsilon = 0.01;
        let pi = std::f32::consts::PI;
        self.phi = self.phi.clamp(epsilon, pi - epsilon);
    }

    pub fn new() -> Self {
        Self {
            theta: 0.0,
            phi: std::f32::consts::FRAC_PI_2,
            radius: 5.0,
            target: Vec3::ZERO,
            sensitivity: 0.005,
            zoom_speed: 1.0,
        }
    }

    pub fn eye(&self) -> Vec3 {
        let sin_phi = self.phi.sin();
        let x = self.radius * sin_phi * self.theta.cos();
        let y = self.radius * self.phi.cos();
        let z = self.radius * sin_phi * self.theta.sin();
        Vec3::new(x, y, z) + self.target
    }

    pub fn handle_mouse_drag(&mut self, delta: Vec2) {
        self.theta -= delta.x * self.sensitivity;
        self.phi = (self.phi - delta.y * self.sensitivity).clamp(0.01, std::f32::consts::PI - 0.01);
    }

    pub fn handle_scroll(&mut self, delta: f32) {
        self.radius = (self.radius - delta * self.zoom_speed).clamp(1.0, 100.0);
    }
}
