use glam::{Mat4, Vec3};

use crate::{
    classes::CelestialBody,
    state::{InstanceData, ModelMatrix},
};

#[derive(Clone, Debug)]
pub struct Star {
    pub metric_info: StarMetricInfo,
    pub properties: StarProperties,
    pub velocity: Vec3,
    pub angular_velocity: Vec3,
    pub rotation_axis: Vec3,
    pub acceleration: Vec3,
    pub ty: StarType,
    pub position: Vec3,
    pub data: InstanceData,
}

impl CelestialBody for Star {
    fn radius(&self) -> f32 {
        self.properties.radius
    }

    fn mass(&self) -> f32 {
        self.metric_info.mass
    }

    fn data(&self) -> crate::state::InstanceData {
        self.data
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

        let new_model = Mat4::from_translation(self.position);
        self.data = InstanceData::new(new_model, self.radius(), self.data.color);
    }

    fn position(&self) -> Vec3 {
        self.position
    }
}

#[derive(Clone, Debug)]
pub struct StarMetricInfo {
    pub mass: f32,
    pub volume: f32,
    pub density: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct StarProperties {
    pub color: Vec3,
    pub temperature: f32,
    pub intensity: f32,
    pub radius: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StarType {
    O, // Blue
    B, // Blue-white
    A, // White
    F, // Yellow-white
    G, // Yellow (like Sun)
    K, // Orange
    M, // Red
    Neutron,
    WhiteDwarf,
    BlackHole,
}

pub fn random_star_type(rng: &mut impl rand::Rng) -> StarType {
    use StarType::*;
    match rng.random_range(0.0..1.0) {
        x if x < 0.00003 => O,
        x if x < 0.0013 => B,
        x if x < 0.006 => A,
        x if x < 0.03 => F,
        x if x < 0.076 => G,
        x if x < 0.121 => K,
        _ => M,
    }
}

impl StarType {
    pub fn properties(&self) -> StarProperties {
        match self {
            StarType::O => StarProperties {
                color: [0.6, 0.8, 1.0].into(), // bluish
                temperature: 30000.0,
                intensity: 100_000.0,
                radius: 6.0,
            },
            StarType::B => StarProperties {
                color: [0.7, 0.85, 1.0].into(),
                temperature: 15000.0,
                intensity: 20_000.0,
                radius: 4.0,
            },
            StarType::A => StarProperties {
                color: [0.9, 0.9, 1.0].into(),
                temperature: 9000.0,
                intensity: 5_000.0,
                radius: 2.5,
            },
            StarType::F => StarProperties {
                color: [1.0, 1.0, 0.9].into(),
                temperature: 7000.0,
                intensity: 2_000.0,
                radius: 1.5,
            },
            StarType::G => StarProperties {
                color: [1.0, 1.0, 0.8].into(),
                temperature: 5800.0,
                intensity: 1.0,
                radius: 1.0,
            },
            StarType::K => StarProperties {
                color: [1.0, 0.8, 0.6].into(),
                temperature: 4500.0,
                intensity: 0.4,
                radius: 0.8,
            },
            StarType::M => StarProperties {
                color: [1.0, 0.6, 0.6].into(),
                temperature: 3200.0,
                intensity: 0.04,
                radius: 0.4,
            },
            StarType::Neutron => StarProperties {
                color: [1.0, 1.0, 1.0].into(),
                temperature: 1_000_000.0,
                intensity: 0.1,
                radius: 0.01,
            },
            StarType::WhiteDwarf => StarProperties {
                color: [0.8, 0.8, 1.0].into(),
                temperature: 8000.0,
                intensity: 0.01,
                radius: 0.1,
            },
            StarType::BlackHole => StarProperties {
                color: [0.0, 0.0, 0.0].into(),
                temperature: 0.0,
                intensity: 0.0,
                radius: 0.2,
            },
        }
    }
}
