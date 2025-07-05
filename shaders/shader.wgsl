struct Camera {
    view_proj: mat4x4<f32>,
    camera_position: vec3<f32>,
    _pad: f32,
};

struct Light {
    position: vec3<f32>,
    _pad1: f32,
    color: vec3<f32>,
    intensity: f32,
};

@group(0) @binding(0)
var<uniform> camera: Camera;

struct InstanceData {
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    radius: f32,
    color: vec3<f32>,
    _pad: f32,
}

@group(1) @binding(0)
var<storage, read> model_matrices: array<InstanceData>;



@group(2) @binding(0)
var<storage, read> lights: array<Light>;

@group(2) @binding(1)
var<uniform> light_count: u32;


struct VSOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) color: vec3<f32>,
};

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @builtin(instance_index) instance_idx: u32,
) -> VSOutput {
    let instance = model_matrices[instance_idx];
    let pos = position * instance.radius; // * vec3(1., -1., 1.)

    let world_pos = instance.normal_matrix * vec4(pos, 1.0);

    // let world_normal = 
    //normalize(
        // (instance.normal_matrix * vec4f(normal, 0.0)).xyz;
    //);
    
    // let normal1 = world_normal;

    let world_normal = (instance.normal_matrix * vec4(normal, 0.0)).xyz;

    var out: VSOutput;
    out.clip_pos = camera.view_proj * world_pos;
    out.normal = world_normal;
    out.world_pos = world_pos.xyz;
    out.color = instance.color;  // Pass through the instance color
    return out;
}

@fragment
fn fs_main(input: VSOutput) -> @location(0) vec4<f32> {
    // Normalize inputs
    let normal = normalize(input.normal);
    let view_dir = normalize(camera.camera_position - input.world_pos);
    
    // Material properties
    let albedo = input.color;
    let shininess = 32.0;
    let ambient_strength = 0.1;
    
    // Initialize lighting
    var total_light = vec3<f32>(ambient_strength) * albedo; // Ambient component

    for (var i = 0u; i < light_count; i++) {
        let light = lights[i];
        
        // Light direction and distance
        let light_vec = light.position - input.world_pos;
        let light_dir = normalize(light_vec);
        let distance = length(light_vec);
        
        // Attenuation (optional)
        let attenuation = 1.0 / (distance * distance);
        
        // Diffuse component
        let diff = max(dot(normal, light_dir), 0.0);
        let diffuse = diff * light.color * light.intensity * attenuation;
        
        // Specular component (Blinn-Phong)
        let halfway_dir = normalize(light_dir + view_dir);
        let spec = pow(max(dot(normal, halfway_dir), 0.0), shininess);
        let specular = spec * light.color * light.intensity * attenuation;
        
        // Combine lighting
        total_light += (diffuse + specular) * albedo;

        // return vec4(normalize(light_vec) * 0.5 + 0.5, 1.0);
    }

    // Gamma correction
    let gamma = 2.2;
    var color = pow(total_light, vec3<f32>(1.0 / gamma));

    // if color.r + color.g + color.b == 0. {
    //     color.r = 1.;
    //     color.b = 1.;
    //     color.g = 1.;
    // }


    return vec4(color, 1.0);
}