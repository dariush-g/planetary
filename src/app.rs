use std::sync::Arc;

use glam::Vec2;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::ActiveEventLoop,
    window::{Window, WindowAttributes, WindowId},
};

use crate::{camera::OrbitCamera, state::State};

#[derive(Debug, Default)]
pub struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
    mouse_pressed: bool,
    last_cursor: Option<Vec2>,
    orbit_camera: OrbitCamera,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(Window::default_attributes().with_title("Planet Renderer"))
            .unwrap();

        self.window = Some(window.into());

        let cloned_window = self.window.clone().unwrap();

        let state = pollster::block_on(State::new(cloned_window));

        self.state = Some(state.into());
        self.last_cursor = None;
        self.mouse_pressed = false;
        self.orbit_camera = OrbitCamera::new();

        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        println!("{event:?}");
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                if self.mouse_pressed {
                    let pos = Vec2::new(position.x as f32, position.y as f32);
                    if let Some(last) = self.last_cursor {
                        let delta = pos - last;

                        if let Some(state) = &mut self.state {
                            state.orbit_camera.rotate(delta);
                        }
                    }
                    self.last_cursor = Some(pos);
                }
            }

            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                self.mouse_pressed = state == ElementState::Pressed;
                if !self.mouse_pressed {
                    self.last_cursor = None;
                }
            }
            WindowEvent::CloseRequested => {
                println!("Close was requested; stopping");
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let Some(state) = &mut self.state {
                    //state.depth_texture = State::create_depth_texture(&state.device, &state.config);

                    state.resize(size);
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(state) = &mut self.state {
                    state.update_camera();
                    state.render().unwrap();
                }
                self.window.clone().unwrap().request_redraw();
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32,
                };

                self.orbit_camera.handle_scroll(scroll);
            }
            _ => (),
        }
    }
}
