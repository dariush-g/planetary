@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<storage> models: array<InstanceData>;
@group(2) @binding(0) var<storage> lights: array<Light>;
@group(2) @binding(1) var<storage> light_count: u32;

@group(3) @binding(0) var g_albedo: texture_2d<f32>;
@group(3) @binding(1) var g_normal: texture_2d<f32>;
@group(3) @binding(2) var g_depth: texture_depth_2d<f32>;

// output texture
@group(3) @binding(3) var output_tex: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let uv = vec2(id.xy) / vec2(textureDimensions(g_albedo));
    let depth = textureLoad(g_depth, id.xy, 0);


    // Reconstruct world pos
    let world_pos = reconstruct_from_depth(uv, depth, camera.view_proj);

    let albedo = textureLoad(g_albedo, id.xy, 0).rgb;
    let normal = normalize(textureLoad(g_normal, id.xy, 0).rgb * 2.0 - 1.0);

    var total_light = vec3(0.1) * albedo;

    for (var i = 0u; i < light_count; i++) {
        let light = lights[i];
        let light_dir = normalize(light.position - world_pos);
        let diff = max(dot(normal, light_dir), 0.0);
        total_light += diff * light.color * light.intensity * albedo;
    }

    textureStore(output_tex, id.xy, vec4(pow(total_light, vec3(1. / 2.2)), 1.0));
}

fn reconstruct_world_pos(uv: vec2f, depth: f32, view_proj: mat4x4<f32>) -> vec3f {
    // Clip space -> NDC
    let clip_pos = vec4(
        uv.x * 2.0 - 1.0,
        1.0 - uv.y * 2.0, // Flip Y for WGSL texture coords
        depth,
        1.0
    );
    
    // NDC -> World
    let world_pos = view_proj * clip_pos;
    return world_pos.xyz / world_pos.w;
}