use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

impl Vertex {
    pub const ATTRS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
    ];

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRS,
        }
    }
}

pub fn generate_uv_sphere(
    lat_segments: u32,
    lon_segments: u32,
    radius: f32,
) -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for y in 0..=lat_segments {
        let theta = std::f32::consts::PI * (y as f32) / (lat_segments as f32);
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        for x in 0..=lon_segments {
            let phi = 2.0 * std::f32::consts::PI * (x as f32) / (lon_segments as f32);
            let sin_phi = phi.sin();
            let cos_phi = phi.cos();

            let nx = cos_phi * sin_theta;
            let ny = cos_theta;
            let nz = sin_phi * sin_theta;

            vertices.push(Vertex {
                position: [nx * radius, ny * radius * -1., nz * radius],
                normal: [nx, ny, nz], // same as position for a sphere
            });
        }
    }

    for y in 0..lat_segments {
        for x in 0..lon_segments {
            let i0 = y * (lon_segments + 1) + x;
            let i1 = i0 + lon_segments + 1;

            indices.extend_from_slice(&[i0, i1, i0 + 1, i1, i1 + 1, i0 + 1]);
        }
    }

    (vertices, indices)
}

