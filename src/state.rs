use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use wgpu::{util::DeviceExt, *};
use winit::{dpi::PhysicalSize, window::Window};

use crate::{
    camera::{Camera, OrbitCamera},
    mesh::{generate_uv_sphere, Vertex},
};

const MAX_SPHERES: usize = 100;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
struct ModelMatrix([[f32; 4]; 4]);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
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
    camera: Camera,
    camera_buffer: Buffer,
    pub orbit_camera: OrbitCamera,
    model_matrices: Vec<ModelMatrix>,
    model_buffer: Buffer,
    model_bind_group: BindGroup,
    pub depth_texture: TextureView,
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

        let (vertices, indices) = generate_uv_sphere(64, 64);

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

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let orbit_camera = OrbitCamera::new();

        let camera = Camera {
            eye: orbit_camera.eye(), //Vec3::new(0., 0., 5.),
            target: Vec3::ZERO,
            up: Vec3::Y,
            aspect: config.width as f32 / config.height as f32,
            fovy: 45.0_f32.to_radians(),
            znear: 0.1,
            zfar: 100.0,
        };

        let camera_uniform = CameraUniform {
            view_proj: camera.build_view_projection_matrix().to_cols_array_2d(),
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
                    visibility: wgpu::ShaderStages::VERTEX,
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

        let model_matrices: Vec<ModelMatrix> = vec![
            ModelMatrix(Mat4::from_translation(Vec3::new(0.0, 0.0, 0.0)).to_cols_array_2d()),
            ModelMatrix(Mat4::from_translation(Vec3::new(3.0, 0.0, 0.0)).to_cols_array_2d()),
            ModelMatrix(Mat4::from_translation(Vec3::new(-3.0, 0.0, 0.0)).to_cols_array_2d()),
            // ... more
        ];

        let model_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Model Matrices"),
            contents: bytemuck::cast_slice(&model_matrices),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let model_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Model Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
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
            bind_group_layouts: &[&camera_bind_group_layout, &model_bind_group_layout],
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
        }
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
                }]),
            );
        }
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
        self.camera.eye = self.orbit_camera.eye();

        let view_proj = self.camera.build_view_projection_matrix();
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[CameraUniform {
                view_proj: view_proj.to_cols_array_2d(),
            }]),
        );

        println!("camera eye: {:?}", self.camera.eye);
    }
}
