use std::sync::Arc;

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    window::{Window, WindowAttributes},
};

#[derive(Debug, Clone, Default)]
pub struct App {
    window: Option<Arc<Window>>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window_attributes = WindowAttributes::default();

        self.window = match event_loop.create_window(window_attributes) {
            Ok(window) => Some(window.into()),
            Err(err) => {
                eprintln!("Error creating window: {err}");
                event_loop.exit();
                return;
            }
        };
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
                println!("{:?}", position)
            }
            WindowEvent::CloseRequested => {
                println!("Close was requested; stopping");
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                self.window
                    .as_ref()
                    .expect("resize event without a window")
                    .request_redraw();
            }
            WindowEvent::RedrawRequested => {
                // Redraw the application.
                //
                // It's preferable for applications that do not render continuously to render in
                // this event rather than in AboutToWait, since rendering in here allows
                // the program to gracefully handle redraws requested by the OS.

                let window = self
                    .window
                    .as_ref()
                    .expect("redraw request without a window");

                // Notify that you're about to draw.
                window.pre_present_notify();

                // Draw.

                // For contiguous redraw loop you can request a redraw from here.
                // window.request_redraw();
            }
            _ => (),
        }
    }
}
