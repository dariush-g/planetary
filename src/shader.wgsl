struct Camera {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: Camera;


struct VSInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) model_index: u32,
};

struct VSOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) normal: vec3<f32>,
};

@group(1) @binding(0)
var<storage, read> model_matrices: array<mat4x4<f32>>;

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @builtin(instance_index) instance_idx: u32,
) -> VSOutput {
    let model = model_matrices[instance_idx];
    let world_pos = model * vec4(position, 1.0);

    var out: VSOutput;
    out.clip_pos = camera.view_proj * world_pos;
    out.normal = normal;
    return out;
}


@fragment
fn fs_main(input: VSOutput) -> @location(0) vec4<f32> {
    let light = normalize(vec3<f32>(0.5, 1.0, 0.7));
    let diffuse = max(dot(input.normal, light), 0.0);
    return vec4(0.1 + diffuse * 0.9, 0.3, 0.9, 1.0);
}
