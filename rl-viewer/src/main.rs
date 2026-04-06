mod protocol;
mod camera;
mod scene;
mod renderer;

use std::sync::Arc;
use std::io::{BufRead, BufReader};
use std::os::unix::net::UnixListener;

use winit::application::ApplicationHandler;
use winit::event::{WindowEvent, MouseButton, MouseScrollDelta, ElementState};
use winit::event_loop::{EventLoop, ActiveEventLoop};
use winit::window::{Window, WindowId};
use winit::keyboard::{KeyCode, PhysicalKey};

use camera::Camera;
use scene::SceneGeometry;
use renderer::Renderer;
use protocol::{Message, FrameData};

const SOCKET_PATH: &str = "/tmp/rl-viewer.sock";

#[derive(Debug)]
enum AppEvent {
    Frame(Box<FrameData>),
    Reset,
    Close,
}

struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    camera: Camera,
    scene: SceneGeometry,
    proxy: winit::event_loop::EventLoopProxy<AppEvent>,
    // Mouse state
    mouse_pressed: bool,
    last_mouse: (f32, f32),
    // Episode tracking
    episode: u32,
    // Recording
    record_path: Option<String>,
}

impl App {
    fn new(proxy: winit::event_loop::EventLoopProxy<AppEvent>, record_path: Option<String>) -> Self {
        let scene = SceneGeometry::new();
        Self {
            window: None,
            renderer: None,
            camera: Camera::default_view(),
            scene,
            proxy,
            mouse_pressed: false,
            last_mouse: (0.0, 0.0),
            episode: 0,
            record_path,
        }
    }

    fn render_frame(&mut self) {
        if let Some(renderer) = &mut self.renderer {
            let vp = self.camera.view_proj();
            match renderer.render(vp) {
                Ok(()) => {}
                Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                    let size = renderer.config.width;
                    let height = renderer.config.height;
                    renderer.resize(winit::dpi::PhysicalSize::new(size, height));
                }
                Err(e) => eprintln!("render error: {e}"),
            }
        }
    }
}

impl ApplicationHandler<AppEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title("RL Viewer — Episode 0")
            .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32));
        let window = Arc::new(event_loop.create_window(attrs).expect("window"));
        self.window = Some(window.clone());

        let static_verts = self.scene.vertices.clone();
        let mut renderer = pollster::block_on(Renderer::new(window.clone(), &static_verts));

        self.camera.set_aspect(renderer.config.width, renderer.config.height);
        if let Some(path) = &self.record_path.clone() {
            renderer.start_recording(path);
        }
        self.renderer = Some(renderer);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(size);
                    self.camera.set_aspect(size.width, size.height);
                }
            }

            WindowEvent::RedrawRequested => {
                self.render_frame();
            }

            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                self.mouse_pressed = state == ElementState::Pressed;
            }

            WindowEvent::CursorMoved { position, .. } => {
                let (x, y) = (position.x as f32, position.y as f32);
                if self.mouse_pressed {
                    let (lx, ly) = self.last_mouse;
                    let dx = x - lx;
                    let dy = y - ly;
                    self.camera.orbit(dx, dy);
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
                self.last_mouse = (x, y);
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.05,
                };
                self.camera.zoom(scroll);
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::Escape) => event_loop.exit(),
                        PhysicalKey::Code(KeyCode::KeyR) => {
                            self.camera = Camera::default_view();
                            if let Some(renderer) = &self.renderer {
                                self.camera.set_aspect(renderer.config.width, renderer.config.height);
                            }
                            if let Some(window) = &self.window {
                                window.request_redraw();
                            }
                        }
                        _ => {}
                    }
                }
            }

            _ => {}
        }
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: AppEvent) {
        match event {
            AppEvent::Frame(frame) => {
                // Soft-follow the first blue car
                let blue_car = frame.cars.iter().find(|c| c.team == 0);
                if let Some(car) = blue_car {
                    let car_pos = car.pos.to_glam();
                    self.camera.follow_car(car_pos);
                }

                // Rebuild dynamic geometry
                if let Some(renderer) = &mut self.renderer {
                    let dyn_verts = self.scene.build_dynamic(&frame);
                    renderer.update_dynamic(&dyn_verts);
                }

                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }

            AppEvent::Reset => {
                self.episode += 1;
                if let Some(window) = &self.window {
                    window.set_title(&format!("RL Viewer — Episode {}", self.episode));
                }
                if let Some(renderer) = &mut self.renderer {
                    renderer.update_dynamic(&[]);
                }
            }

            AppEvent::Close => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.stop_recording();
                }
                _event_loop.exit();
            }
        }
    }
}

fn spawn_socket_listener(proxy: winit::event_loop::EventLoopProxy<AppEvent>) {
    std::thread::spawn(move || {
        // Remove stale socket file
        let _ = std::fs::remove_file(SOCKET_PATH);

        let listener = match UnixListener::bind(SOCKET_PATH) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("Failed to bind socket {SOCKET_PATH}: {e}");
                return;
            }
        };

        eprintln!("Listening on {SOCKET_PATH}");

        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    let proxy = proxy.clone();
                    std::thread::spawn(move || {
                        let reader = BufReader::new(stream);
                        for line in reader.lines() {
                            let line = match line {
                                Ok(l) => l,
                                Err(_) => break,
                            };
                            if line.is_empty() { continue; }

                            match serde_json::from_str::<Message>(&line) {
                                Ok(Message::Frame(frame)) => {
                                    let _ = proxy.send_event(AppEvent::Frame(Box::new(frame)));
                                }
                                Ok(Message::Reset) => {
                                    let _ = proxy.send_event(AppEvent::Reset);
                                }
                                Ok(Message::Close) => {
                                    let _ = proxy.send_event(AppEvent::Close);
                                    return;
                                }
                                Err(e) => {
                                    eprintln!("JSON parse error: {e}");
                                }
                            }
                        }
                    });
                }
                Err(e) => eprintln!("Accept error: {e}"),
            }
        }
    });
}

fn main() {
    env_logger::init();

    // Parse --record <output_path>
    let args: Vec<String> = std::env::args().collect();
    let record_path = args.iter()
        .position(|a| a == "--record")
        .and_then(|i| args.get(i + 1))
        .cloned();

    let event_loop = EventLoop::<AppEvent>::with_user_event()
        .build()
        .expect("event loop");

    let proxy = event_loop.create_proxy();
    spawn_socket_listener(proxy.clone());

    let mut app = App::new(proxy, record_path);
    event_loop.run_app(&mut app).expect("run");
}
