use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use wgpu::util::DeviceExt;

use pollster::FutureExt as _;

use cgmath::prelude::*;

use rand::prelude::*;
mod texture;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

struct ToggleKey {
    was_down: bool,
}
impl ToggleKey {
    pub fn new() -> Self {
        Self { was_down: false }
    }
    fn down(&mut self, state: bool) -> bool {
        if !self.was_down && state {
            self.was_down = true;
            return true;
        } else if !state {
            self.was_down = false;
        }
        false
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
}
impl Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            }],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Cell {
    position: [f32; 3],
    hp: i32,
    neighbors: i32,
}
impl Cell {
    fn new(position: [f32; 3], hp: i32) -> Self {
        Self {
            position,
            hp,
            neighbors: 0,
        }
    }
    fn get_color(&self) -> [f32; 3] {
        if self.hp == STATE {
            ALIVE_COLOR
        } else {
            let intensity = (1. + self.hp as f32) / (STATE as f32);
            [
                rgb_to_srgb(
                    intensity * (DYING_COLOR[0] - DEAD_COLOR[0]) as f32 + DEAD_COLOR[0] as f32,
                ),
                rgb_to_srgb(
                    intensity * (DYING_COLOR[1] - DEAD_COLOR[1]) as f32 + DEAD_COLOR[1] as f32,
                ),
                rgb_to_srgb(
                    intensity * (DYING_COLOR[2] - DEAD_COLOR[2]) as f32 + DEAD_COLOR[2] as f32,
                ),
            ]
        }
    }
    fn create_instance(&self) -> Instance {
        Instance {
            position: self.position,
            color: self.get_color(),
        }
    }
    fn should_draw(&self) -> bool {
        self.hp >= 0
    }
}

#[derive(Clone, Copy)]
struct Instance {
    position: [f32; 3],
    color: [f32; 3],
}
impl Instance {
    fn to_raw(&self) -> InstanceRaw {
        InstanceRaw {
            model: cgmath::Matrix4::from_translation(self.position.into()).into(),
            color: self.color,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
    model: [[f32; 4]; 4],
    color: [f32; 3],
}
impl InstanceRaw {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    //  color
                    offset: std::mem::size_of::<[f32; 16]>() as wgpu::BufferAddress,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

const VERTICES: &[Vertex] = &[
    Vertex {
        // A - top left
        position: [-0.5, 0.5, 0.],
    },
    Vertex {
        // B - bottom left
        position: [-0.5, -0.5, 0.],
    },
    Vertex {
        // C - bottom right
        position: [0.5, -0.5, 0.],
    },
    Vertex {
        // D - top right
        position: [0.5, 0.5, 0.],
    },
    Vertex {
        // E - top right - back
        position: [0.5, 0.5, -1.],
    },
    Vertex {
        // F - bottom right - back
        position: [0.5, -0.5, -1.],
    },
    Vertex {
        // G - top left - back
        position: [-0.5, 0.5, -1.],
    },
    Vertex {
        // H - bottom left - back
        position: [-0.5, -0.5, -1.],
    },
];
const INDICES: &[u16] = &[
    0, 1, 2, 0, 2, 3, // front face
    3, 2, 5, 3, 5, 4, // right face
    4, 5, 7, 4, 7, 6, // back face
    6, 7, 1, 6, 1, 0, // left face
    6, 0, 3, 6, 3, 4, // top face
    2, 1, 7, 5, 2, 7, // bottom face
];

const CELL_BOUNDS: u32 = 96;
const INSTANCE_DISPLACEMENT: [f32; 3] = [
    CELL_BOUNDS as f32 * 0.5,
    CELL_BOUNDS as f32 * 0.5,
    CELL_BOUNDS as f32 * 0.5,
];

pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0,
);

const STATE: i32 = 10;
const SURVIVAL: [i32; 27] = [
    0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];
const SPAWN: [i32; 27] = [
    0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];
const ALIVE_CHANCE_ON_START: f32 = 0.15;
const CLEAR_COLOR: wgpu::Color = wgpu::Color {
    r: 0.023104,
    g: 0.030257,
    b: 0.047776,
    a: 1.,
};
const ALIVE_COLOR: [f32; 3] = [0.529523, 0.119264, 0.144972];
const DEAD_COLOR: [u8; 3] = [76, 86, 106];
const DYING_COLOR: [u8; 3] = [216, 222, 233];
const TEXT_COLOR: [f32; 4] = [0.84337, 0.867136, 0.907547, 1.];

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Rules {
    survival: [Wrapped; 27],
    spawn: [Wrapped; 27],
    state: i32,
    cell_bounds: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Wrapped {
    x: i32,
    padding0: i32,
    padding1: i32,
    padding2: i32,
}
fn rgb_to_srgb(rgb: f32) -> f32 {
    (rgb as f32 / 255.).powf(2.2)
}

struct Camera {
    eye: cgmath::Point3<f32>,
    target: cgmath::Point3<f32>,
    up: cgmath::Vector3<f32>,
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,

    lat: f64,
    lon: f64,
    radius: f64,

    turn_speed: f64,
    zoom_speed: f64,
    drag_speed: f64,

    up_down: bool,
    down_down: bool,
    left_down: bool,
    right_down: bool,
    forward_down: bool,
    backward_down: bool,

    lmb_down: bool,
    last_mouse_pos: (f64, f64),
}
impl Camera {
    fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        let view = cgmath::Matrix4::look_at_rh(self.eye, self.target, self.up);
        let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);
        return proj * view;
    }
    fn process_events(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state,
                        virtual_keycode: Some(keycode),
                        ..
                    },
                ..
            } => {
                let pressed = *state == ElementState::Pressed;
                match keycode {
                    VirtualKeyCode::W | VirtualKeyCode::Up => {
                        self.up_down = pressed;
                        true
                    }
                    VirtualKeyCode::S | VirtualKeyCode::Down => {
                        self.down_down = pressed;
                        true
                    }
                    VirtualKeyCode::A | VirtualKeyCode::Left => {
                        self.left_down = pressed;
                        true
                    }
                    VirtualKeyCode::D | VirtualKeyCode::Right => {
                        self.right_down = pressed;
                        true
                    }
                    VirtualKeyCode::Q | VirtualKeyCode::PageUp => {
                        self.forward_down = pressed;
                        true
                    }
                    VirtualKeyCode::E | VirtualKeyCode::PageDown => {
                        self.backward_down = pressed;
                        true
                    }
                    _ => false,
                }
            }
            &WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                self.lmb_down = state == ElementState::Pressed;
                true
            }
            &WindowEvent::CursorMoved { position, .. } => {
                if self.lmb_down {
                    let (x, y) = (position.x, position.y);
                    let (dx, dy) = (x - self.last_mouse_pos.0, y - self.last_mouse_pos.1);
                    self.lon -= dx * self.drag_speed;
                    self.lat += dy * self.drag_speed;
                    self.update_eye();
                    self.last_mouse_pos = (position.x, position.y);
                }
                self.last_mouse_pos = (position.x, position.y);
                true
            }
            _ => false,
        }
    }
    fn update(&mut self, delta: f64) {
        if self.up_down {
            self.lat += self.turn_speed * delta;
        }
        if self.down_down {
            self.lat -= self.turn_speed * delta;
        }
        if self.left_down {
            self.lon -= self.turn_speed * delta;
        }
        if self.right_down {
            self.lon += self.turn_speed * delta;
        }
        if self.forward_down {
            self.radius -= self.zoom_speed * delta;
        }
        if self.backward_down {
            self.radius += self.zoom_speed * delta;
        }
        self.clamp_pos();
        self.update_eye();
    }
    fn clamp_pos(&mut self) {
        if self.lat > std::f64::consts::PI / 2. {
            self.lat = std::f64::consts::PI / 2. - 0.0001;
        } else if self.lat < -std::f64::consts::PI / 2. {
            self.lat = -std::f64::consts::PI / 2. + 0.0001;
        }
        if self.lon > std::f64::consts::PI {
            self.lon -= 2. * std::f64::consts::PI;
        } else if self.lon < -std::f64::consts::PI {
            self.lon += 2. * std::f64::consts::PI;
        }
        if self.radius < 5. {
            self.radius = 5.;
        }
    }
    fn update_eye(&mut self) {
        self.eye = cgmath::Point3::new(
            (self.radius * self.lat.cos() * self.lon.cos()) as f32,
            (self.radius * self.lat.cos() * self.lon.sin()) as f32,
            (self.radius * self.lat.sin()) as f32,
        );
    }
}

struct CameraStaging {
    camera: Camera,
}
impl CameraStaging {
    fn new(camera: Camera) -> Self {
        Self { camera }
    }
    fn update_camera(&self, camera_uniform: &mut CameraUniform) {
        camera_uniform.view_proj =
            (OPENGL_TO_WGPU_MATRIX * self.camera.build_view_projection_matrix()).into();
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    transform: [[f32; 4]; 4],
}
impl CameraUniform {
    fn new() -> Self {
        Self {
            view_proj: cgmath::Matrix4::identity().into(),
            transform: cgmath::Matrix4::identity().into(),
        }
    }
}

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,

    brush:
        wgpu_text::TextBrush<wgpu_text::font::FontRef<'static>, glyph_brush::DefaultSectionHasher>,

    camera_staging: CameraStaging,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,

    depth_texture: texture::Texture,
    render_pipeline: wgpu::RenderPipeline,

    compute_storage_buffer: wgpu::Buffer,
    compute_staging_buffer: wgpu::Buffer,
    compute_bind_group: wgpu::BindGroup,

    sync_compute_pipeline: wgpu::ComputePipeline,
    cn_compute_pipeline: wgpu::ComputePipeline,

    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,

    instance_data: Vec<InstanceRaw>,
    instance_buffer: wgpu::Buffer,

    last_frame: f64,
    delta: f64,
    ticks: u64,
    backend: wgpu::Backend,

    p_tk: ToggleKey,
    rmb_tk: ToggleKey,

    paused: bool,

    scissor_rect: (u32, u32, u32, u32),

    cells: Vec<Cell>,
}
impl State {
    async fn new(window: &Window) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let surface = unsafe { instance.create_surface(window) };

        println!("\nBackends list: {:#?}", wgpu::Backends::all());
        let adapter = match instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
        {
            Some(adapter) => adapter,
            None => {
                cfg_if::cfg_if! {
                    if #[cfg(target_arch = "wasm32")] {
                        panic!("No adapter found")
                    } else {
                        instance
                            .enumerate_adapters(wgpu::Backends::all())
                            .filter(|adapter| surface.get_supported_formats(&adapter).len() > 0)
                            .next()
                            .unwrap()
                    }
                }
            }
        };
        let backend = adapter.get_info().backend;
        println!("Current backend:: {:#?}", backend);
        println!(
            "Supported texture formats: {:#?}\n",
            surface.get_supported_formats(&adapter)
        );

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                    label: None,
                },
                None,
            )
            .await
            .unwrap();

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_supported_formats(&adapter)[0],
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        surface.configure(&device, &config);

        let mut camera = Camera {
            eye: (1., 1., 1.).into(),
            target: (0., 0., 0.).into(),
            up: cgmath::Vector3::unit_z(),
            aspect: size.width as f32 / size.height as f32,
            fovy: 45.,
            znear: 0.01,
            zfar: CELL_BOUNDS as f32 * 5.,
            lat: 0.35,
            lon: 0.35,
            radius: CELL_BOUNDS as f64 * 2.5,
            turn_speed: std::f64::consts::PI / 4.,
            zoom_speed: CELL_BOUNDS as f64 / 4.,
            drag_speed: 1. / 200.,
            up_down: false,
            down_down: false,
            left_down: false,
            right_down: false,
            forward_down: false,
            backward_down: false,
            lmb_down: false,
            last_mouse_pos: (0., 0.),
        };
        camera.update_eye();

        let mut camera_uniform = CameraUniform::new();
        let camera_staging = CameraStaging::new(camera);
        camera_staging.update_camera(&mut camera_uniform);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                label: Some("camera_bind_group_layout"),
            });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        let depth_texture =
            texture::Texture::create_depth_texture(&device, &config, "depth_texture");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout],
                push_constant_ranges: &[],
            });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc(), InstanceRaw::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: texture::Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });
        let num_indices = INDICES.len() as u32;

        let cells: Vec<Cell> = Self::create_cells();
        let instance_data: Vec<InstanceRaw> = cells
            .iter()
            .map(|cell| cell.create_instance().to_raw())
            .collect();

        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("compute.wgsl").into()),
        });
        let compute_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compute Staging Buffer"),
            size: cells.len() as u64 * std::mem::size_of::<Cell>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let compute_storage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Compute Storage Buffer"),
            contents: bytemuck::cast_slice(&cells),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let sync_compute_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Compute Pipeline"),
                layout: None,
                module: &compute_shader,
                entry_point: "sync",
            });
        let cn_compute_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Compute Pipeline"),
                layout: None,
                module: &compute_shader,
                entry_point: "count_neighbors",
            });
        let sync_compute_bind_group_layout = sync_compute_pipeline.get_bind_group_layout(0);

        let mut rules = Rules {
            survival: [Wrapped {
                x: 0,
                padding0: 0,
                padding1: 0,
                padding2: 0,
            }; 27],
            spawn: [Wrapped {
                x: 0,
                padding0: 0,
                padding1: 0,
                padding2: 0,
            }; 27],
            state: STATE,
            cell_bounds: CELL_BOUNDS,
        };
        for i in 0..27 {
            rules.survival[i].x = SURVIVAL[i];
            rules.spawn[i].x = SPAWN[i];
        }

        let compute_rules_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Compute Rules Buffer"),
            contents: bytemuck::cast_slice(&[rules]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: &sync_compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: compute_rules_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: compute_storage_buffer.as_entire_binding(),
                },
            ],
        });

        let font: &[u8] = include_bytes!("../assets/PixelSquare.ttf");
        let brush = wgpu_text::BrushBuilder::using_font_bytes(font)
            .unwrap()
            .build(&device, &config);

        let last_frame = instant::now();
        let delta = 0.2;
        let ticks = 0;

        let p_tk = ToggleKey::new();
        let rmb_tk = ToggleKey::new();

        let paused = false;

        let scissor_rect = (0, 0, size.width, size.height);

        Self {
            surface,
            device,
            queue,
            config,
            size,
            brush,
            camera_staging,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            depth_texture,
            render_pipeline,
            compute_staging_buffer,
            compute_storage_buffer,
            compute_bind_group,
            sync_compute_pipeline,
            cn_compute_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            instance_data,
            instance_buffer,
            last_frame,
            delta,
            ticks,
            backend,
            p_tk,
            rmb_tk,
            paused,
            scissor_rect,
            cells,
        }
    }
    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture =
                texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
            self.brush.resize_view(
                self.config.width as f32,
                self.config.height as f32,
                &self.queue,
            );

            let width = std::cmp::min(self.size.width, (self.size.height * 16) / 9);
            let height = std::cmp::min(self.size.height, (self.size.width * 9) / 16);
            let x = (self.size.width - width) / 2;
            let y = (self.size.height - height) / 2;
            self.scissor_rect = (x, y, width, height);
            self.camera_uniform.transform[0][0] = width as f32 / self.size.width as f32;
            self.camera_uniform.transform[1][1] = height as f32 / self.size.height as f32;
        }
    }
    fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state,
                        virtual_keycode: Some(keycode),
                        ..
                    },
                ..
            } => {
                let pressed = *state == ElementState::Pressed;
                match keycode {
                    VirtualKeyCode::R => {
                        if pressed {
                            self.cells = Self::create_cells();
                            self.ticks = 0;
                            self.queue.write_buffer(
                                &self.compute_storage_buffer,
                                0,
                                bytemuck::cast_slice(&self.cells),
                            );
                        }
                    }
                    VirtualKeyCode::Space => {
                        if pressed {
                            self.camera_staging.camera.lat = 0.35;
                            self.camera_staging.camera.lon = 0.35;
                            self.camera_staging.camera.radius = CELL_BOUNDS as f64 * 2.5;
                            self.camera_staging.camera.update_eye();
                        }
                    }
                    VirtualKeyCode::P => {
                        if self.p_tk.down(pressed) {
                            self.paused = !self.paused;
                        }
                    }
                    _ => {}
                }
            }
            &WindowEvent::MouseInput {
                state,
                button: MouseButton::Right,
                ..
            } => {
                if self.rmb_tk.down(state == ElementState::Pressed) {
                    self.paused = !self.paused;
                }
            }
            _ => {}
        }
        self.camera_staging.camera.process_events(event)
    }
    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.calc_instance_data();
        self.queue.write_buffer(
            &self.instance_buffer,
            0,
            bytemuck::cast_slice(&self.instance_data),
        );

        self.camera_staging.camera.update(self.delta);
        self.camera_staging.update_camera(&mut self.camera_uniform);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        if !self.paused {
            self.ticks += 1;
            {
                let mut cn_compute_pass =
                    encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
                cn_compute_pass.set_pipeline(&self.cn_compute_pipeline);
                cn_compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);
                cn_compute_pass.insert_debug_marker("compute count neighbors");
                cn_compute_pass.dispatch_workgroups(CELL_BOUNDS, CELL_BOUNDS, CELL_BOUNDS);
            }
            {
                let mut sync_compute_pass =
                    encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
                sync_compute_pass.set_pipeline(&self.sync_compute_pipeline);
                sync_compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);
                sync_compute_pass.insert_debug_marker("compute sync");
                sync_compute_pass.dispatch_workgroups(CELL_BOUNDS, CELL_BOUNDS, CELL_BOUNDS);
            }
            encoder.copy_buffer_to_buffer(
                &self.compute_storage_buffer,
                0,
                &self.compute_staging_buffer,
                0,
                self.cells.len() as u64 * std::mem::size_of::<Cell>() as wgpu::BufferAddress,
            );
        }

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(CLEAR_COLOR),
                        store: true,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });

            render_pass.set_scissor_rect(
                self.scissor_rect.0,
                self.scissor_rect.1,
                self.scissor_rect.2,
                self.scissor_rect.3,
            );

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..self.instance_data.len() as u32)
        }

        let fps_str = format!("FPS: {:.0}\n", 1. / self.delta);
        let ticks_str = format!("Ticks: {}\n", self.ticks);
        let backend_str = format!("Backend: {:?}\n", self.backend);
        let font_size = self.scissor_rect.2 as f32 / 75.;

        let mut section = glyph_brush::Section::default()
            .with_screen_position((
                self.scissor_rect.0 as f32 + self.scissor_rect.2 as f32 / 100.,
                self.scissor_rect.1 as f32 + self.scissor_rect.2 as f32 / 100.,
            ))
            .add_text(
                glyph_brush::Text::new(&fps_str[..])
                    .with_color(TEXT_COLOR)
                    .with_scale(font_size),
            )
            .add_text(
                glyph_brush::Text::new(&ticks_str[..])
                    .with_color(TEXT_COLOR)
                    .with_scale(font_size),
            )
            .add_text(
                glyph_brush::Text::new(&backend_str)
                    .with_color(TEXT_COLOR)
                    .with_scale(font_size),
            )
            .with_layout(
                glyph_brush::Layout::default_wrap().h_align(glyph_brush::HorizontalAlign::Left),
            );
        if self.paused {
            section = section.add_text(
                glyph_brush::Text::new("PAUSED")
                    .with_color(TEXT_COLOR)
                    .with_scale(font_size),
            );
        }
        self.brush.queue(&section);

        let text_buffer = self.brush.draw(&self.device, &view, &self.queue);

        self.queue.submit([encoder.finish(), text_buffer]);
        output.present();

        if !self.paused {
            let compute_buffer_slice = self.compute_staging_buffer.slice(..);
            let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
            compute_buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

            self.device.poll(wgpu::Maintain::Wait);

            if let Some(Ok(())) = receiver.receive().block_on() {
                let data = compute_buffer_slice.get_mapped_range();
                self.cells = bytemuck::cast_slice(&data).to_vec();

                drop(data);
                self.compute_staging_buffer.unmap();
            } else {
                panic!("GPU compute failled")
            }
        }

        self.delta = (instant::now() - self.last_frame) / 1000.;
        self.last_frame = instant::now();

        Ok(())
    }

    fn calc_instance_data(&mut self) {
        self.instance_data.clear();
        for cell in self.cells.iter() {
            // TODO: possibly remove this loop
            if cell.should_draw() {
                self.instance_data.push(cell.create_instance().to_raw());
            }
        }
    }
    fn create_cells() -> Vec<Cell> {
        let mut rng = rand::thread_rng();
        let mut cells = Vec::new();
        for x in 0..CELL_BOUNDS {
            for y in 0..CELL_BOUNDS {
                for z in 0..CELL_BOUNDS {
                    let mut cell = Cell::new(
                        [
                            x as f32 - INSTANCE_DISPLACEMENT[0],
                            y as f32 - INSTANCE_DISPLACEMENT[1],
                            z as f32 - INSTANCE_DISPLACEMENT[2],
                        ],
                        -1,
                    );
                    if x >= CELL_BOUNDS / 3
                        && x <= CELL_BOUNDS * 2 / 3
                        && y >= CELL_BOUNDS / 3
                        && y <= CELL_BOUNDS * 2 / 3
                        && z >= CELL_BOUNDS / 3
                        && z <= CELL_BOUNDS * 2 / 3
                        && ALIVE_CHANCE_ON_START < rng.gen()
                    {
                        cell.hp = STATE;
                    }
                    cells.push(cell);
                }
            }
        }
        return cells;
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init_with_level(log::Level::Warn).expect("Couldn't initialize logger");
        } else {
            env_logger::init();
        }
    }

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_inner_size(winit::dpi::PhysicalSize::new(1200, 675))
        .with_title("3d Cellular Automata [WGPU/Rust]")
        .with_resizable(true)
        .build(&event_loop)
        .unwrap();

    #[cfg(target_arch = "wasm32")]
    {
        window.set_inner_size(winit::dpi::PhysicalSize::new(800, 450));

        use winit::platform::web::WindowExtWebSys;
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                let dst = doc.get_element_by_id("body")?;
                let canvas = web_sys::Element::from(window.canvas());
                dst.append_child(&canvas).ok()?;
                Some(())
            })
            .expect("Couldn't add canvas to doc");
    }

    let mut state = State::new(&window).await;

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window.id() => {
            if !state.input(event) {
                match event {
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    } => *control_flow = ControlFlow::Exit,
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        state.resize(**new_inner_size);
                    }
                    _ => {}
                }
            }
        }
        Event::RedrawRequested(window_id) if window_id == window.id() => match state.render() {
            Ok(_) => {}
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                state.resize(state.size)
            }
            Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
            Err(wgpu::SurfaceError::Timeout) => log::warn!("Surface timeout"),
        },
        Event::RedrawEventsCleared => {
            window.request_redraw();
        }
        _ => {}
    });
}
