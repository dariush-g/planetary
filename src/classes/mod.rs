use dyn_clone::DynClone;
use glam::Vec3;

use crate::state::InstanceData;

pub mod planet;
pub mod star;

pub trait CelestialBody: DynClone + std::fmt::Debug {
    fn radius(&self) -> f32;
    fn mass(&self) -> f32;
    fn data(&self) -> InstanceData;
    fn axis(&self) -> Vec3;
    fn velocity(&self) -> Vec3;
    fn angular_velocity(&self) -> Vec3;
    fn acceleration(&self) -> Vec3;
    fn apply_force(&mut self, force: Vec3);
    fn update(&mut self);
    fn position(&self) -> Vec3;
    fn reset_forces(&mut self);
    fn add_force(&mut self, force: Vec3);
}

dyn_clone::clone_trait_object!(CelestialBody);
