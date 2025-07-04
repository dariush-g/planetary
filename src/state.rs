use crate::classes::CelestialBody;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use rayon::iter::IntoParallelRefMutIterator;
use std::{fmt, sync::Arc, time::*};
use wgpu::{util::DeviceExt, *};
use winit::{dpi::PhysicalSize, window::Window};

use crate::{
    camera::{Camera, OrbitCamera},
    mesh::{generate_uv_sphere, Vertex},
};

const MAX_SPHERES: usize = 100;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
pub struct InstanceData {
    model: ModelMatrix,
    // normal_matrix: ModelMatrix,
    color: [f32; 3],
    _pad: f32,
}

impl InstanceData {
    pub fn new(model: ModelMatrix, color: [f32; 3]) -> Self {
        Self {
            model,
            color,
            _pad: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
pub struct ModelMatrix(pub [[f32; 4]; 4]);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    camera_position: [f32; 3],
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct LightSource {
    position: [f32; 3],
    _pad1: f32,
    color: [f32; 3],
    intensity: f32,
}

struct TimerCallback(Box<dyn FnMut() + Send + 'static>);

// Manually implement Debug since closures can't derive it
impl fmt::Debug for TimerCallback {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TimerCallback")
    }
}

#[derive(Debug)]
pub struct State {
    surface: Arc<Surface<'static>>,
    pub device: Device,
    queue: Queue,
    pub config: SurfaceConfiguration,
    surface_format: TextureFormat,
    size: PhysicalSize<u32>,
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    index_count: u32,
    render_pipeline: RenderPipeline,
    camera_bind_group: BindGroup,
    pub camera: Camera,
    camera_buffer: Buffer,
    pub orbit_camera: OrbitCamera,
    pub model_matrices: Vec<InstanceData>,
    model_buffer: Buffer,
    model_bind_group: BindGroup,
    pub depth_texture: TextureView,
    lights: Vec<LightSource>,
    light_bind_group: BindGroup,
    light_buffer: Buffer,
    start_time: Instant,
    last_frame_time: Instant,
    accumulated_time: Duration,
    timers: Vec<(Duration, TimerCallback)>,
    pub bodies: Option<Vec<Box<dyn CelestialBody>>>,
    // g_buffer: GBufferTextures,
    // lighting_pipeline: ComputePipeline,
    // lighting_bind_group: BindGroup,
}

#[derive(Debug, Clone)]
struct GBufferTextures {
    albedo: TextureView,
    normal: TextureView,
    depth: TextureView,
    output: TextureView,
}

impl State {
    pub async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();

        let instance = Instance::default();
        let surface = Arc::new(instance.create_surface(window).unwrap());

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                required_features: Features::empty(),
                required_limits: Limits::default(),
                label: None,
                ..Default::default()
            })
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats[0];

        let config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 0,
        };

        surface.configure(&device, &config);

        let (vertices, indices) = generate_uv_sphere(128, 128, 2.);

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let index_count = indices.len() as u32;

        let shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/shader.wgsl"));

        let orbit_camera = OrbitCamera::new();

        let camera = Camera {
            eye: orbit_camera.eye(), //Vec3::new(0., 0., 5.),
            target: Vec3::ZERO,
            up: Vec3::Y,
            aspect: config.width as f32 / config.height as f32,
            fovy: 45.0_f32.to_radians(),
            znear: 0.1,
            zfar: 10_000.0,
        };

        let camera_uniform = CameraUniform {
            view_proj: camera.build_view_projection_matrix().to_cols_array_2d(),
            camera_position: orbit_camera.eye().into(),
            _pad: 0.0,
        };

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Camera Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("Camera Bind Group"),
        });

        let matrices: Vec<ModelMatrix> = vec![
            ModelMatrix(Mat4::from_translation(Vec3::new(0.0, 7.5, 0.0)).to_cols_array_2d()),
            ModelMatrix(Mat4::from_translation(Vec3::new(3.0, 0.0, 0.0)).to_cols_array_2d()),
            ModelMatrix(Mat4::from_translation(Vec3::new(-3.0, -1.0, 0.0)).to_cols_array_2d()),
            // ModelMatrix(Mat4::from_translation(Vec3::new(0.0, 4.0, 0.0)).to_cols_array_2d()),
        ];

        let model_matrices: Vec<InstanceData> = (0..matrices.len())
            .map(|i| {
                InstanceData {
                    model: matrices[i],
                    // normal_matrix: ModelMatrix(
                    //     Mat4::from_cols_array_2d(&matrices[i].0)
                    //         .inverse()
                    //         .transpose()
                    //         .to_cols_array_2d(),
                    // ),
                    color: match i {
                        0 => [1.0, 0.0, 0.0], // Red
                        1 => [0.0, 1.0, 0.0], // Green
                        2 => [0.0, 0.0, 1.0], // Blue
                        _ => [1.0, 1.0, 0.0], // Yellow (fallback)
                    },
                    _pad: 0.0,
                }
            })
            .collect();

        // let model_matrices: Vec<ModelMatrix> = (0..30)
        //     .flat_map(|i| {
        //         (0..30).flat_map(move |j| {
        //             (0..30).map(move |k| {
        //                 ModelMatrix(
        //                     Mat4::from_translation(Vec3::new(
        //                         (i * 3) as f32,
        //                         (j * 3) as f32,
        //                         (k * 3) as f32,
        //                     ))
        //                     .to_cols_array_2d(),
        //                 )
        //             })
        //         })
        //     })
        //     .collect();

        let lights = [
            LightSource {
                position: [0., 4., 0.],
                _pad1: 0.,
                color: [1., 1., 1.],
                intensity: 10.,
            },
            // LightSource {
            //     position: [0., 0., 0.],
            //     _pad1: 0.,
            //     color: [1., 1., 1.],
            //     intensity: 10.,
            // },
        ];

        let light_count_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Light Count Uniform Buffer"),
            contents: bytemuck::bytes_of(&lights.len()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Light Storage Buffer"),
            contents: bytemuck::cast_slice(&lights),
            usage: wgpu::BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        let light_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Light Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &light_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: light_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: light_count_buffer.as_entire_binding(),
                },
            ],
            label: Some("Model Bind Group"),
        });

        let model_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Model Matrices"),
            contents: bytemuck::cast_slice(&model_matrices),
            usage: wgpu::BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        let model_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Model Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let model_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &model_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: model_buffer.as_entire_binding(),
            }],
            label: Some("Model Bind Group"),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[
                &camera_bind_group_layout,
                &model_bind_group_layout,
                &light_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Planet Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                front_face: wgpu::FrontFace::Ccw,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // let shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/lighting.wgsl"));

        // let compute_pipeline_layout =
        //     device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        //         label: Some("Compute Pipeline Layout"),
        //         bind_group_layouts: &[
        //             &camera_bind_group_layout,
        //             &model_bind_group_layout,
        //             &light_bind_group_layout,
        //         ],
        //         push_constant_ranges: &[],
        //     });

        let depth_texture = Self::create_depth_texture(&device, &config);

        Self {
            surface,
            device,
            queue,
            config,
            surface_format,
            size,
            vertex_buffer,
            index_buffer,
            index_count,
            render_pipeline,
            camera_bind_group,
            camera_buffer,
            camera,
            orbit_camera,
            model_buffer,
            model_matrices,
            model_bind_group,
            depth_texture,
            lights: lights.to_vec(),
            light_bind_group,
            light_buffer,
            start_time: Instant::now(),
            last_frame_time: Instant::now(),
            accumulated_time: Duration::ZERO,
            timers: Vec::new(),
            bodies: None,
        }
    }

    pub fn add_timer(&mut self, delay_ms: u64, callback: impl FnMut() + Send + 'static) {
        self.timers.push((
            Duration::from_millis(delay_ms),
            TimerCallback(Box::new(callback)),
        ));
    }

    pub fn update_timers(&mut self) {
        let now = Instant::now();
        let delta = now - self.last_frame_time;
        self.last_frame_time = now;
        self.accumulated_time += delta;

        self.timers.retain_mut(|(remaining, callback)| {
            *remaining = remaining.saturating_sub(delta);
            if remaining.is_zero() {
                (callback.0)();
                false
            } else {
                true
            }
        });
    }

    pub fn create_depth_texture(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
    ) -> wgpu::TextureView {
        let size = wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        };

        let desc = wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        };

        let texture = device.create_texture(&desc);
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);

            self.depth_texture = Self::create_depth_texture(&self.device, &self.config);

            self.camera.aspect = self.config.width as f32 / self.config.height as f32;

            let view_proj = self.camera.build_view_projection_matrix();
            self.queue.write_buffer(
                &self.camera_buffer,
                0,
                bytemuck::cast_slice(&[CameraUniform {
                    view_proj: view_proj.to_cols_array_2d(),
                    camera_position: self.camera.eye.into(),
                    _pad: 0.0,
                }]),
            );
        }
    }
    pub fn add_model(&mut self, trans: Vec3, color: [f32; 3]) {
        let model = ModelMatrix(Mat4::from_translation(trans).to_cols_array_2d());
        self.model_matrices.push(InstanceData::new(model, color));

        let buffer_size = self.model_matrices.len() * std::mem::size_of::<InstanceData>();

        if buffer_size as u64 > self.model_buffer.size() {
            self.model_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Model Buffer"),
                size: buffer_size as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let model_bind_group_layout =
                self.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("Model Bind Group Layout"),
                        entries: &[wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        }],
                    });

            self.model_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &model_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.model_buffer.as_entire_binding(),
                }],
                label: Some("Model Bind Group"),
            });
        }

        // 5. Update buffer contents
        self.queue.write_buffer(
            &self.model_buffer,
            0,
            bytemuck::cast_slice(&self.model_matrices),
        );

        // 6. Make sure your render pipeline uses the correct instance count
        // This would be used in your draw call later
    }
    pub fn render(&mut self) -> Result<(), SurfaceError> {
        let frame = self.surface.get_current_texture()?;
        let view = frame.texture.create_view(&TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let color_attachment = wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: StoreOp::Store,
                },
            };

            let depth_attachment = wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_texture,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            };

            let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("Planet Render Pass"),
                color_attachments: &[Some(color_attachment)],
                depth_stencil_attachment: Some(depth_attachment),
                ..Default::default()
            });

            rpass.set_bind_group(0, &self.camera_bind_group, &[]);
            rpass.set_bind_group(1, &self.model_bind_group, &[]);
            rpass.set_bind_group(2, &self.light_bind_group, &[]);

            rpass.set_pipeline(&self.render_pipeline);

            //for i in 0..self.model_matrices.len() as u32 {
            rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            rpass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);

            rpass.draw_indexed(0..self.index_count, 0, 0..self.model_matrices.len() as u32);
            //}
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();

        Ok(())
    }

    pub fn update_camera(&mut self) {
        let target_eye = self.orbit_camera.eye();
        self.update_lights();
        self.camera.eye = self.camera.eye.lerp(target_eye, 0.05); // 20% interpolation

        let view_proj = self.camera.build_view_projection_matrix();
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[CameraUniform {
                view_proj: view_proj.to_cols_array_2d(),
                camera_position: self.camera.eye.into(),
                _pad: 0.0,
            }]),
        );

        // println!("camera eye: {:?}", self.camera.eye);
    }

    pub fn update_lights(&mut self) {
        println!("{:?}", self.lights[0].position);
        self.queue
            .write_buffer(&self.light_buffer, 0, bytemuck::cast_slice(&self.lights));
    }

    pub fn update_model_position(&mut self, index: usize, new_position: Vec3) {
        if index >= self.model_matrices.len() {
            return; // Safety check
        }

        let new_model = Mat4::from_translation(new_position);
        self.model_matrices[index].model = ModelMatrix(new_model.to_cols_array_2d());

        self.queue.write_buffer(
            &self.model_buffer,
            0,
            bytemuck::cast_slice(&self.model_matrices),
        );
    }

    pub fn update_n_model_positions(&mut self, updates: &[(usize, Vec3)]) {
        for (index, pos) in updates {
            if *index >= self.model_matrices.len() {
                continue;
            }

            let new_model = Mat4::from_translation(*pos);
            self.model_matrices[*index].model = ModelMatrix(new_model.to_cols_array_2d());
        }

        self.queue.write_buffer(
            &self.model_buffer,
            0,
            bytemuck::cast_slice(&self.model_matrices),
        );
    }

    pub fn animate_models(&mut self, delta_time: f32) {
        let time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f32();

        for (i, instance) in self.model_matrices.iter_mut().enumerate() {
            let offset = i as f32 * 3.0;
            let new_pos = Vec3::new(offset + time.sin(), (time * 0.5).cos(), 0.0);

            let new_model = Mat4::from_translation(new_pos);
            instance.model = ModelMatrix(new_model.to_cols_array_2d());
        }

        self.queue.write_buffer(
            &self.model_buffer,
            0,
            bytemuck::cast_slice(&self.model_matrices),
        );
    }

    pub fn move_model_relative(&mut self, index: usize, offset: Vec3) {
        if index >= self.model_matrices.len() {
            return;
        }

        let current_model = Mat4::from_cols_array_2d(&self.model_matrices[index].model.0);
        let current_translation = current_model.w_axis.truncate();

        let new_translation = current_translation + offset;

        let new_model = Mat4::from_translation(new_translation);

        // 4. Update both matrices
        self.model_matrices[index].model = ModelMatrix(new_model.to_cols_array_2d());

        // 5. Single buffer update
        self.queue.write_buffer(
            &self.model_buffer,
            0,
            bytemuck::cast_slice(&self.model_matrices),
        );
    }

    pub fn apply_matrix(&mut self, index: usize, mat: Mat4) {
        if index >= self.model_matrices.len() {
            return;
        }

        let current_model = Mat4::from_cols_array_2d(&self.model_matrices[index].model.0);
        let new_model = mat * current_model; // Apply new transform first

        self.model_matrices[index].model.0 = new_model.to_cols_array_2d();

        self.queue.write_buffer(
            &self.model_buffer,
            0,
            bytemuck::cast_slice(&self.model_matrices),
        );
    }

    pub fn apply_veloc(&mut self, bodies: &mut Vec<Box<dyn CelestialBody>>) {
        let indices = bodies.len();
        let bodies_clone = dyn_clone::clone(bodies);

        for i in 0..indices {
            let body_i = &mut bodies[i];
            // F = G * M1 * M2 / |R||^2 ) * r (normalized)
            //
            for index in 0..indices {
                let body_j = bodies_clone[index].clone();
                if body_i.position() != body_j.position() {
                    compute_force(body_i, &body_j);
                }
            }
        }
    }

    pub fn update_body_positions(&mut self) {}
}

fn compute_force(
    body_i: &mut Box<dyn CelestialBody + 'static>,
    body_j: &Box<dyn CelestialBody + 'static>,
) {
    let mut force = body_i.acceleration();

    let r = body_i.position() - body_j.position();
    let r_norm = r.length();

    force += (body_i.mass() * body_j.mass() / (r * r)) * r_norm;

    body_i.apply_force(force);
}
