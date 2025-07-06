use glam::{Mat4, Vec3};
use planetary::{
    app::App,
    classes::{planet::*, star::*, *},
    state::*,
};
use winit::event_loop::EventLoop;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;

    let bodies: Vec<Box<dyn CelestialBody>> = vec![
        Box::new(Planet {
            metric_info: PlanetMetricInfo {
                mass: 1.0e3,
                radius: 5.,
                volume: 10.,
                density: 10.,
            },
            ty: PlanetType::Rocky,
            position: Vec3::new(0., 0., 450.),
            velocity: Vec3::new(3., 0., 1.),
            rotation: Vec3::ZERO,
            acceleration: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            rotation_axis: Vec3::Y,
            data: InstanceData::new(
                Mat4::from_translation(Vec3::new(0., 0., 450.)),
                1.,
                [0., 0., 128.],
            ),
            accumulated_force: Vec3::ZERO,
        }),
        Box::new(Planet {
            metric_info: PlanetMetricInfo {
                mass: 1.0e2,
                radius: 1.,
                volume: 10.,
                density: 10.,
            },
            ty: PlanetType::Rocky,
            position: Vec3::new(350., 0., 0.),
            velocity: Vec3::new(0., 0.5, 1.),
            rotation: Vec3::ZERO,
            acceleration: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            rotation_axis: Vec3::Y,
            data: InstanceData::new(
                Mat4::from_translation(Vec3::new(350., 0., 0.)),
                1.,
                [1., 1., 1.],
            ),
            accumulated_force: Vec3::ZERO,
        }),
        Box::new(Star {
            metric_info: StarMetricInfo {
                mass: 1.0e11,
                volume: 10.,
                density: 10.,
            },
            properties: StarProperties {
                color: Vec3::new(1.0, 0., 0.),
                temperature: 10.,
                intensity: 10.,
                radius: 100.,
            },
            velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            rotation_axis: Vec3::Y,
            acceleration: Vec3::ZERO,
            ty: StarType::A,
            position: Vec3::ZERO,
            data: InstanceData::new(Mat4::from_translation(Vec3::ZERO), 10., [1., 0., 0.]),
            accumulated_force: Vec3::ZERO,
        }),
    ];

    env_logger::Builder::from_default_env().init();

    let mut app = App::new(Some(bodies));

    event_loop.run_app(&mut app)?;
    Ok(())
}
