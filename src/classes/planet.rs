use glam::Vec3;

use crate::{classes::CelestialBody, state::ModelMatrix};

#[derive(Clone, Debug)]
pub struct Planet {
    pub metric_info: PlanetMetricInfo,
    pub ty: PlanetType,
    pub position: Vec3,
    pub velocity: Vec3,
    pub rotation: Vec3,
    pub acceleration: Vec3,
    pub angular_velocity: Vec3,
    pub rotation_axis: Vec3,
    pub model: ModelMatrix,
    pub color: Vec3,
}

impl CelestialBody for Planet {
    fn radius(&self) -> f32 {
        self.metric_info.radius
    }

    fn mass(&self) -> f32 {
        self.metric_info.mass
    }

    fn model(&self) -> ModelMatrix {
        self.model
    }

    fn color(&self) -> Vec3 {
        self.color
    }

    fn position(&self) -> Vec3 {
        self.position
    }

    fn axis(&self) -> Vec3 {
        self.rotation_axis
    }

    fn velocity(&self) -> Vec3 {
        self.velocity
    }

    fn angular_velocity(&self) -> Vec3 {
        self.angular_velocity
    }

    fn acceleration(&self) -> Vec3 {
        self.acceleration
    }

    fn apply_force(&mut self, force: Vec3) {
        self.acceleration += force;
    }

    fn update(&mut self) {
        self.velocity += self.acceleration;
        self.position += self.velocity;
        self.acceleration = Vec3::ZERO;
    }
}

#[derive(Clone, Debug)]
pub struct PlanetMetricInfo {
    pub mass: f32,
    pub radius: f32,
    pub volume: f32,
    pub density: f32,
}

#[derive(Clone, Debug)]
pub enum PlanetType {
    Rocky,
    GasGiant,
    IceGiant,
    Dwarf,
    Lava,
    Ocean,
    Desert,
}

impl PlanetType {
    pub fn material_properties(&self) -> Material {
        match self {
            PlanetType::Rocky => Material::new([0.4, 0.3, 0.2], 0.1, 0.8),
            PlanetType::GasGiant => Material::new([0.9, 0.7, 0.2], 0.0, 0.5),
            PlanetType::IceGiant => Material::new([0.6, 0.8, 1.0], 0.0, 0.2),
            PlanetType::Dwarf => Material::new([0.5, 0.5, 0.5], 0.1, 0.6),
            PlanetType::Lava => Material::new([1.0, 0.2, 0.0], 0.5, 0.9),
            PlanetType::Ocean => Material::new([0.0, 0.4, 0.7], 0.0, 0.1),
            PlanetType::Desert => Material::new([0.9, 0.7, 0.4], 0.0, 0.2),
        }
    }
}

pub struct Material {
    pub albedo: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,
}

impl Material {
    pub fn new(albedo: [f32; 3], metallic: f32, roughness: f32) -> Self {
        Self {
            albedo,
            metallic,
            roughness,
        }
    }
}
