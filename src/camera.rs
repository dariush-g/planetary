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

    pub fn pan(&mut self, delta: Vec2, speed: f32) {
        let right = (self.target - self.eye).cross(self.up).normalize();
        let up = self.up.normalize();

        let world_right = right * (-delta.x * speed);
        let world_up = up * (delta.y * speed);

        self.eye += world_right + world_up;
        self.target += world_right + world_up;
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
    last_operation: Option<CameraOperation>,
}

#[derive(Clone, Debug)]
enum CameraOperation {
    Rotate,
    Pan,
    Zoom,
}

impl OrbitCamera {
    pub fn apply_to_camera(&self, camera: &mut Camera) {
        camera.eye = self.eye();
        camera.target = self.target;
        //camera.up = Vec3::Y;
    }

    pub fn rotate(&mut self, delta: glam::Vec2) {
        if let Some(CameraOperation::Pan) = self.last_operation {
            self.update_angles_from_eye();
        }

        self.last_operation = Some(CameraOperation::Rotate);

        self.theta -= delta.x * self.sensitivity;
        self.phi -= delta.y * self.sensitivity;

        // Clamp phi to avoid flipping
        self.phi = self.phi.clamp(0.01, std::f32::consts::PI - 0.01);
    }

    pub fn update_angles_from_eye(&mut self) {
        let offset = self.eye() - self.target;
        self.radius = offset.length();

        if self.radius > 0.0 {
            self.phi = (offset.y / self.radius).acos();
            self.theta = offset.z.atan2(offset.x);

            // Normalize theta to avoid accumulating large values
            self.theta = self.theta.rem_euclid(2.0 * std::f32::consts::PI);
        }
    }

    pub fn new() -> Self {
        Self {
            theta: 0.0,
            phi: std::f32::consts::FRAC_PI_2,
            radius: 5.0,
            target: Vec3::ZERO,
            sensitivity: 0.005,
            zoom_speed: 1.0,
            last_operation: None,
        }
    }

    fn get_forward(&self) -> Vec3 {
        let sin_phi = self.phi.sin();
        let cos_phi = self.phi.cos();
        let sin_theta = self.theta.sin();
        let cos_theta = self.theta.cos();

        Vec3::new(sin_phi * cos_theta, cos_phi, sin_phi * sin_theta).normalize()
    }

    pub fn pan(&mut self, mut delta: Vec2) {
        delta.x *= -1.;

        self.last_operation = Some(CameraOperation::Pan);

        let pan_speed = self.radius * 0.002;
        let forward = self.get_forward();
        let right = forward.cross(Vec3::Y).normalize();
        let up = right.cross(forward).normalize();

        self.target += right * (-delta.x * pan_speed) + up * (delta.y * pan_speed);
    }

    pub fn eye(&self) -> Vec3 {
        let sin_phi = self.phi.sin();
        let cos_phi = self.phi.cos();
        let sin_theta = self.theta.sin();
        let cos_theta = self.theta.cos();

        let x = self.radius * sin_phi * cos_theta;
        let y = self.radius * cos_phi;
        let z = self.radius * sin_phi * sin_theta;

        glam::Vec3::new(x, y, z) + self.target
    }

    pub fn handle_mouse_drag(&mut self, delta: Vec2) {
        self.theta -= delta.x * self.sensitivity;
        self.phi = (self.phi - delta.y * self.sensitivity).clamp(0.01, std::f32::consts::PI - 0.01);
    }

    pub fn handle_scroll(&mut self, delta: f32) {
        self.last_operation = Some(CameraOperation::Zoom);

        let zoom_amount = delta * self.zoom_speed;
        self.radius -= zoom_amount;

        self.radius = self.radius.clamp(1., 100.);
        println!("{}", self.radius);
    }
}
