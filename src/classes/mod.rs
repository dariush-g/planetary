use glam::Vec3;

use crate::state::ModelMatrix;

pub mod planet;
pub mod star;

pub trait CelestialBody {
    fn radius(&self) -> f32;
    fn mass(&self) -> f32;
    fn model(&self) -> ModelMatrix;
    fn color(&self) -> Vec3;
    fn position(&self) -> Vec3;
}
