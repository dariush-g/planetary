pub mod animation;
pub mod app;
pub mod camera;
pub mod classes;
pub mod mesh;
pub mod oct;
pub mod state;
use glam::{Mat4, Vec3};
use winit::event_loop::EventLoop;

use crate::{
    app::App,
    classes::{
        planet::{Planet, PlanetMetricInfo},
        CelestialBody,
    },
    state::{InstanceData, ModelMatrix},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;

    let bodies: Vec<Box<dyn CelestialBody>> = vec![
        Box::new(Planet {
            metric_info: PlanetMetricInfo {
                mass: 100.,
                radius: 1.,
                volume: 10.,
                density: 10.,
            },
            ty: classes::planet::PlanetType::Rocky,
            position: Vec3::ZERO,
            velocity: Vec3::new(0., 0., 0.),
            rotation: Vec3::ZERO,
            acceleration: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            rotation_axis: Vec3::Y,
            data: InstanceData::new(
                ModelMatrix(Mat4::from_translation(Vec3::ZERO).to_cols_array_2d()),
                1.,
                [1., 1., 1.],
            ),
        }),
        Box::new(Planet {
            metric_info: PlanetMetricInfo {
                mass: 10.,
                radius: 1.,
                volume: 10.,
                density: 10.,
            },
            ty: classes::planet::PlanetType::Rocky,
            position: Vec3::new(0., 2., 0.),
            velocity: Vec3::new(0., 0., 0.),
            rotation: Vec3::ZERO,
            acceleration: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            rotation_axis: Vec3::Y,
            data: InstanceData::new(
                ModelMatrix(Mat4::from_translation(Vec3::new(2., 0., 0.)).to_cols_array_2d()),
                1.,
                [0., 1., 1.],
            ),
        }),
    ];

    event_loop.run_app(&mut App::new(Some(bodies)))?;
    Ok(())
}
