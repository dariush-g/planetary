struct Camera {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: Camera;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct VSOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) normal: vec3<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VSOutput {
    var out: VSOutput;
    out.clip_pos = camera.view_proj * vec4(input.position, 1.0);
    out.normal = input.normal;
    return out;
}

@fragment
fn fs_main(input: VSOutput) -> @location(0) vec4<f32> {
    let light = normalize(vec3<f32>(0.5, 1.0, 0.7));
    let diffuse = max(dot(input.normal, light), 0.0);
    return vec4(0.1 + diffuse * 0.9, 0.3, 0.9, 1.0);
}
