struct CameraUniform {
    view_proj: mat4x4<f32>,
    transform: mat4x4<f32>,
}
@group(0) @binding(0)
var<uniform> camera: CameraUniform;


struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) idx: u32,
}

struct InstanceInput {
    @location(5) model_matrix_0: vec4<f32>,
    @location(6) model_matrix_1: vec4<f32>,
    @location(7) model_matrix_2: vec4<f32>,
    @location(8) model_matrix_3: vec4<f32>,
    @location(9) color: vec3<f32>,
    @location(10) light_top_left: f32,
    @location(11) light_bottom_left: f32,
    @location(12) light_bottom_right: f32,
    @location(13) light_top_right: f32,
    @location(14) light_top_right_back: f32,
    @location(15) light_bottom_right_back: f32,
    @location(16) light_top_left_back: f32,
    @location(17) light_bottom_left_back: f32,
}


struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};


@vertex
fn vs_main(model: VertexInput, instance: InstanceInput) -> VertexOutput {
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );
    var out: VertexOutput;
    var lighting: f32;
    if (model.idx == 0u) {
        lighting = instance.light_top_left;
    } else if (model.idx == 1u) {
        lighting = instance.light_bottom_left;
    } else if (model.idx == 2u) {
        lighting = instance.light_bottom_right;
    } else if (model.idx == 3u) {
        lighting = instance.light_top_right;
    } else if (model.idx == 4u) {
        lighting = instance.light_top_right_back;
    } else if (model.idx == 5u) {
        lighting = instance.light_bottom_right_back;
    } else if (model.idx == 6u) {
        lighting = instance.light_top_left_back;
    } else if (model.idx == 7u) {
        lighting = instance.light_bottom_left_back;
    }
    out.color = instance.color * lighting;
    out.clip_position = camera.view_proj * model_matrix * vec4<f32>(model.position, 1.0);
    out.clip_position = out.clip_position * camera.transform;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4(in.color, 1.0);
}