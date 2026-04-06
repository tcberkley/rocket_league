use std::io::Write;
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use crate::scene::LineVertex;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
}

pub struct Renderer {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    uniform_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    static_vertex_buf: wgpu::Buffer,
    static_vertex_count: u32,
    dynamic_vertex_buf: wgpu::Buffer,
    dynamic_vertex_count: u32,
    dynamic_capacity: u32,
    // Recording
    record_stdin: Option<std::process::ChildStdin>,
    record_process: Option<std::process::Child>,
}

impl Renderer {
    pub async fn new(
        window: std::sync::Arc<winit::window::Window>,
        static_verts: &[LineVertex],
    ) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let surface = instance.create_surface(window).expect("create surface");
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("adapter");
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .expect("device");

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats.iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            // COPY_SRC lets us read back pixels for recording
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Depth texture
        let (depth_texture, depth_view) = make_depth_texture(&device, config.width, config.height);

        // Uniforms
        let uniform_data = Uniforms { view_proj: Mat4::IDENTITY.to_cols_array_2d() };
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniforms"),
            contents: bytemuck::bytes_of(&uniform_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        // Shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("line_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/line.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let vertex_buf_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<LineVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("line_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[vertex_buf_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Static vertex buffer
        let static_vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("static_vb"),
            contents: bytemuck::cast_slice(static_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // Dynamic vertex buffer (pre-allocated, overwritten each frame)
        let dynamic_capacity = 8192u32;
        let dynamic_vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dynamic_vb"),
            size: (dynamic_capacity as usize * std::mem::size_of::<LineVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            surface, device, queue, config, pipeline,
            uniform_buf, bind_group,
            depth_texture, depth_view,
            static_vertex_buf,
            static_vertex_count: static_verts.len() as u32,
            dynamic_vertex_buf,
            dynamic_vertex_count: 0,
            dynamic_capacity,
            record_stdin: None,
            record_process: None,
        }
    }

    /// Start piping raw BGRA frames to ffmpeg → `output_path`.
    /// Requires ffmpeg on PATH. No-op if ffmpeg is missing.
    pub fn start_recording(&mut self, output_path: &str) {
        use std::process::{Command, Stdio};
        let w = self.config.width;
        let h = self.config.height;
        // Detect pixel format: Metal surfaces are Bgra8UnormSrgb
        let pix_fmt = if self.config.format.is_srgb() { "bgra" } else { "rgba" };
        eprintln!("[record] starting → {output_path}  ({w}x{h} {pix_fmt})");
        let result = Command::new("ffmpeg")
            .args([
                "-y",
                "-f", "rawvideo",
                "-pixel_format", pix_fmt,
                "-video_size", &format!("{w}x{h}"),
                "-framerate", "15",
                "-i", "pipe:0",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18",
                output_path,
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn();

        match result {
            Ok(mut child) => {
                self.record_stdin = child.stdin.take();
                self.record_process = Some(child);
            }
            Err(e) => {
                eprintln!("[record] failed to spawn ffmpeg: {e}");
                eprintln!("[record] install with: brew install ffmpeg");
            }
        }
    }

    /// Flush and finalize the recording. Blocks until ffmpeg exits.
    pub fn stop_recording(&mut self) {
        self.record_stdin.take(); // dropping stdin closes the pipe → ffmpeg finalizes
        if let Some(mut child) = self.record_process.take() {
            let _ = child.wait();
            eprintln!("[record] done");
        }
    }

    pub fn is_recording(&self) -> bool {
        self.record_stdin.is_some()
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 { return; }
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
        (self.depth_texture, self.depth_view) = make_depth_texture(
            &self.device, self.config.width, self.config.height,
        );
    }

    pub fn update_dynamic(&mut self, verts: &[LineVertex]) {
        let count = verts.len().min(self.dynamic_capacity as usize);
        if count == 0 { self.dynamic_vertex_count = 0; return; }
        let bytes = bytemuck::cast_slice(&verts[..count]);
        self.queue.write_buffer(&self.dynamic_vertex_buf, 0, bytes);
        self.dynamic_vertex_count = count as u32;
    }

    pub fn render(&mut self, view_proj: Mat4) -> Result<(), wgpu::SurfaceError> {
        // Upload uniforms
        let uniforms = Uniforms { view_proj: view_proj.to_cols_array_2d() };
        self.queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&Default::default());

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.03, g: 0.06, b: 0.03, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);

            // Static geometry (field)
            if self.static_vertex_count > 0 {
                pass.set_vertex_buffer(0, self.static_vertex_buf.slice(..));
                pass.draw(0..self.static_vertex_count, 0..1);
            }

            // Dynamic geometry (ball, cars, pads)
            if self.dynamic_vertex_count > 0 {
                pass.set_vertex_buffer(0, self.dynamic_vertex_buf.slice(..));
                pass.draw(0..self.dynamic_vertex_count, 0..1);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        // Capture frame before presenting (while texture is still valid)
        if self.record_stdin.is_some() {
            self.capture_frame(&output);
        }

        output.present();
        Ok(())
    }

    fn capture_frame(&mut self, output: &wgpu::SurfaceTexture) {
        let w = self.config.width;
        let h = self.config.height;
        // Row stride must be a multiple of 256 (wgpu requirement)
        let bytes_per_row = align256(w * 4);
        let buf_size = (bytes_per_row * h) as u64;

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("capture_staging"),
            size: buf_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            output.texture.as_image_copy(),
            wgpu::ImageCopyBuffer {
                buffer: &staging,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.queue.submit(std::iter::once(encoder.finish()));

        // Map synchronously
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        let data = slice.get_mapped_range();
        let stdin = match self.record_stdin.as_mut() {
            Some(s) => s,
            None => return,
        };

        // Write row-by-row, stripping the alignment padding
        for row in 0..h {
            let start = (row * bytes_per_row) as usize;
            let end = start + (w * 4) as usize;
            if stdin.write_all(&data[start..end]).is_err() {
                break;
            }
        }
    }
}

fn align256(x: u32) -> u32 {
    (x + 255) & !255
}

fn make_depth_texture(device: &wgpu::Device, w: u32, h: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&Default::default());
    (tex, view)
}
