pub mod app;
pub mod mesh;
pub mod state;
pub mod camera;

use winit::event_loop::EventLoop;

use crate::app::App;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;

    event_loop.run_app(&mut App::default())?;
    Ok(())
}
