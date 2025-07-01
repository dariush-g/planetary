use std::sync::Arc;

use glam::Vec2;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowAttributes, WindowId},
};

use crate::{
    camera::{Camera, OrbitCamera},
    state::State,
};

#[derive(Debug, Default)]
pub struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,

    left_mouse_pressed: bool,
    right_mouse_pressed: bool,

    last_cursor: Option<Vec2>,
    //orbit_camera: OrbitCamera,
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
        self.left_mouse_pressed = false;
        self.right_mouse_pressed = false;
        //self.orbit_camera = OrbitCamera::new();

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
            WindowEvent::KeyboardInput { event, .. } => match event.physical_key {
                PhysicalKey::Code(KeyCode::KeyR) => {
                    if let Some(state) = &mut self.state {
                        state.orbit_camera = OrbitCamera::new();
                        //state.camera = Camera::default();
                    }
                }
                _ => {}
            },
            WindowEvent::CursorMoved { position, .. } => {
                let new_pos = Vec2::new(position.x as f32, position.y as f32);

                if let Some(old_pos) = self.last_cursor {
                    let delta = new_pos - old_pos;

                    if delta.length() < 0.5 {
                        // Ignore movements smaller than 0.5 pixels
                        return;
                    }

                    if let Some(state) = &mut self.state {
                        match (self.left_mouse_pressed, self.right_mouse_pressed) {
                            (true, false) => {
                                state.orbit_camera.rotate(-delta);
                                state.update_camera();
                            }
                            (false, true) => {
                                state.orbit_camera.pan(delta);
                                state.orbit_camera.apply_to_camera(&mut state.camera);
                                state.update_camera();
                            }
                            _ => {}
                        }
                    }
                }

                self.last_cursor = Some(new_pos);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;

                match button {
                    MouseButton::Left => {
                        self.left_mouse_pressed = pressed;
                        if pressed {
                            // When left button is pressed, ensure right button is not pressed
                            self.right_mouse_pressed = false;
                        }
                        if !pressed {
                            self.last_cursor = None;
                        }
                    }
                    MouseButton::Right => {
                        self.right_mouse_pressed = pressed;
                        if pressed {
                            // When right button is pressed, ensure left button is not pressed
                            self.left_mouse_pressed = false;
                        }
                        if !pressed {
                            self.last_cursor = None;
                        }
                    }
                    _ => {}
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

                if let Some(state) = &mut self.state {
                    state.orbit_camera.handle_scroll(scroll);
                }
            }

            _ => (),
        }
    }
}
