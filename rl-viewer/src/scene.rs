use crate::protocol::{CarData, FrameData, rotmat_to_glam};
use glam::{Mat4, Vec3};

// RL field constants (uu)
pub const SIDE_WALL_X: f32 = 4096.0;
pub const BACK_WALL_Y: f32 = 5120.0;
pub const CEILING_Z: f32 = 2044.0;
pub const GOAL_HALF_WIDTH: f32 = 892.755;
pub const GOAL_HEIGHT: f32 = 642.775;
pub const GOAL_DEPTH: f32 = 880.0;
pub const BALL_RADIUS: f32 = 91.25;

// Car hitbox approximate dimensions (uu)
const CAR_LEN: f32 = 118.0;
const CAR_WIDTH: f32 = 84.0;
const CAR_HEIGHT: f32 = 36.0;

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct LineVertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
}

pub struct SceneGeometry {
    pub vertices: Vec<LineVertex>,
}

impl SceneGeometry {
    pub fn new() -> Self {
        let mut v = Vec::new();
        build_field(&mut v);
        Self { vertices: v }
    }

    /// Rebuild dynamic geometry (ball, cars) from a frame
    pub fn build_dynamic(&self, frame: &FrameData) -> Vec<LineVertex> {
        let mut v = Vec::new();
        build_ball(&mut v, frame);
        for car in &frame.cars {
            build_car(&mut v, car);
        }
        build_boost_pads(&mut v, &frame.boost_pads);
        v
    }
}

// ---- helpers ----------------------------------------------------------------

fn line(v: &mut Vec<LineVertex>, a: Vec3, b: Vec3, color: [f32; 4]) {
    v.push(LineVertex { position: a.into(), color });
    v.push(LineVertex { position: b.into(), color });
}

fn rect_lines(v: &mut Vec<LineVertex>, corners: [Vec3; 4], color: [f32; 4]) {
    line(v, corners[0], corners[1], color);
    line(v, corners[1], corners[2], color);
    line(v, corners[2], corners[3], color);
    line(v, corners[3], corners[0], color);
}

fn box_lines(v: &mut Vec<LineVertex>, min: Vec3, max: Vec3, color: [f32; 4]) {
    let corners = [
        Vec3::new(min.x, min.y, min.z),
        Vec3::new(max.x, min.y, min.z),
        Vec3::new(max.x, max.y, min.z),
        Vec3::new(min.x, max.y, min.z),
        Vec3::new(min.x, min.y, max.z),
        Vec3::new(max.x, min.y, max.z),
        Vec3::new(max.x, max.y, max.z),
        Vec3::new(min.x, max.y, max.z),
    ];
    // Bottom face
    for i in 0..4 { line(v, corners[i], corners[(i+1)%4], color); }
    // Top face
    for i in 0..4 { line(v, corners[i+4], corners[(i+1)%4+4], color); }
    // Verticals
    for i in 0..4 { line(v, corners[i], corners[i+4], color); }
}

fn circle_lines(v: &mut Vec<LineVertex>, center: Vec3, radius: f32, normal_axis: u8, color: [f32; 4], segments: u32) {
    let step = std::f32::consts::TAU / segments as f32;
    for i in 0..segments {
        let a = i as f32 * step;
        let b = (i + 1) as f32 * step;
        let (ax, ay) = (a.cos() * radius, a.sin() * radius);
        let (bx, by) = (b.cos() * radius, b.sin() * radius);
        let (pa, pb) = match normal_axis {
            0 => (Vec3::new(center.x, center.y + ax, center.z + ay),
                  Vec3::new(center.x, center.y + bx, center.z + by)),
            1 => (Vec3::new(center.x + ax, center.y, center.z + ay),
                  Vec3::new(center.x + bx, center.y, center.z + by)),
            _ => (Vec3::new(center.x + ax, center.y + ay, center.z),
                  Vec3::new(center.x + bx, center.y + by, center.z)),
        };
        line(v, pa, pb, color);
    }
}

// ---- static field geometry --------------------------------------------------

fn build_field(v: &mut Vec<LineVertex>) {
    let gray: [f32; 4] = [0.4, 0.4, 0.4, 1.0];
    let green: [f32; 4] = [0.1, 0.45, 0.15, 1.0];
    let white: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
    let orange: [f32; 4] = [1.0, 0.55, 0.1, 1.0];
    let blue: [f32; 4] = [0.2, 0.5, 1.0, 1.0];

    // Map RL coords (x,y,z) -> render coords (x, z, -y)
    // Floor outline
    let fx = SIDE_WALL_X;
    let fy = BACK_WALL_Y;
    let fz = 0.0_f32;
    rect_lines(v, [
        Vec3::new(-fx, fz, -fy),
        Vec3::new( fx, fz, -fy),
        Vec3::new( fx, fz,  fy),
        Vec3::new(-fx, fz,  fy),
    ], green);

    // Grid lines on floor
    for xi in [-2048, 0, 2048] {
        let x = xi as f32;
        line(v, Vec3::new(x, fz, -fy), Vec3::new(x, fz, fy), [0.15, 0.35, 0.1, 0.5]);
    }
    for yi in [-2560, 0, 2560] {
        let y = yi as f32;
        line(v, Vec3::new(-fx, fz, y), Vec3::new(fx, fz, y), [0.15, 0.35, 0.1, 0.5]);
    }

    // Center line
    line(v, Vec3::new(-fx, fz, 0.0), Vec3::new(fx, fz, 0.0), white);
    // Center circle (r=600, on floor)
    circle_lines(v, Vec3::new(0.0, fz, 0.0), 600.0, 1, white, 32);

    // Side walls
    let ceil = CEILING_Z;
    for &x in &[-fx, fx] {
        rect_lines(v, [
            Vec3::new(x, 0.0,  -fy),
            Vec3::new(x, 0.0,   fy),
            Vec3::new(x, ceil,  fy),
            Vec3::new(x, ceil, -fy),
        ], gray);
    }

    // Back walls
    for &(y_sign, color) in &[(-1.0_f32, blue), (1.0_f32, orange)] {
        let y_rl = y_sign * fy;
        let y_r = -y_rl; // render y
        rect_lines(v, [
            Vec3::new(-fx, 0.0,  y_r),
            Vec3::new( fx, 0.0,  y_r),
            Vec3::new( fx, ceil, y_r),
            Vec3::new(-fx, ceil, y_r),
        ], gray);

        // Goal box (extends toward center)
        let gw = GOAL_HALF_WIDTH;
        let gh = GOAL_HEIGHT;
        let gd = GOAL_DEPTH * y_sign;
        box_lines(v,
            Vec3::new(-gw, 0.0, y_r),
            Vec3::new( gw, gh,  y_r + gd),
            color,
        );
    }
}

// ---- dynamic geometry -------------------------------------------------------

fn build_ball(v: &mut Vec<LineVertex>, frame: &FrameData) {
    let white: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
    let center = frame.ball.pos.to_glam();
    let r = BALL_RADIUS;
    // Three circles (longitude/latitude) on ball
    for axis in 0..3 {
        circle_lines(v, center, r, axis, white, 16);
    }
}

fn build_car(v: &mut Vec<LineVertex>, car: &CarData) {
    let color: [f32; 4] = if car.team == 0 {
        [0.3, 0.6, 1.0, 1.0]  // blue
    } else {
        [1.0, 0.55, 0.1, 1.0] // orange
    };

    let pos = car.pos.to_glam();
    let rot = rotmat_to_glam(&car.rot);

    // Car box corners in local space
    let hx = CAR_LEN / 2.0;
    let hy = CAR_WIDTH / 2.0;
    let hz = CAR_HEIGHT / 2.0;

    let local = [
        Vec3::new(-hx, -hy, -hz),
        Vec3::new( hx, -hy, -hz),
        Vec3::new( hx,  hy, -hz),
        Vec3::new(-hx,  hy, -hz),
        Vec3::new(-hx, -hy,  hz),
        Vec3::new( hx, -hy,  hz),
        Vec3::new( hx,  hy,  hz),
        Vec3::new(-hx,  hy,  hz),
    ];

    let world: Vec<Vec3> = local.iter()
        .map(|&lp| pos + rot * lp)
        .collect();

    // Bottom face
    for i in 0..4 { line(v, world[i], world[(i+1)%4], color); }
    // Top face
    for i in 0..4 { line(v, world[i+4], world[(i+1)%4+4], color); }
    // Verticals
    for i in 0..4 { line(v, world[i], world[i+4], color); }

    // Forward arrow
    let forward = rot * Vec3::new(hx + 60.0, 0.0, 0.0);
    let tip = [1.0, 1.0, 0.3, 1.0];
    line(v, pos, pos + forward, tip);
}

fn build_boost_pads(v: &mut Vec<LineVertex>, timers: &[f32]) {
    // 34 standard boost pad positions in RL (x, y) — major pads first
    let pads: &[(f32, f32)] = &[
        // Big pads (6)
        (-3072.0, -4096.0), (3072.0, -4096.0),
        (-3584.0,  0.0),    (3584.0,  0.0),
        (-3072.0,  4096.0), (3072.0,  4096.0),
        // Small pads (28)
        (0.0, -4240.0), (-1792.0, -4184.0), (1792.0, -4184.0),
        (-940.0, -3308.0), (940.0, -3308.0),
        (0.0, -2816.0), (-3584.0, -2484.0), (3584.0, -2484.0),
        (-1788.0, -2300.0), (1788.0, -2300.0),
        (-2048.0,  -1036.0), (0.0, -1024.0), (2048.0, -1036.0),
        (-1024.0, 0.0), (1024.0, 0.0),
        (-2048.0,   1036.0), (0.0,  1024.0), (2048.0,  1036.0),
        (-1788.0,  2300.0), (1788.0,  2300.0),
        (-3584.0,  2484.0), (3584.0,  2484.0),
        (-940.0,  3308.0), (940.0,  3308.0),
        (0.0,  2816.0), (-1792.0,  4184.0), (1792.0,  4184.0),
        (0.0,  4240.0),
    ];

    for (i, &(px, py)) in pads.iter().enumerate() {
        let available = i >= timers.len() || timers[i] <= 0.0;
        let color: [f32; 4] = if available {
            [1.0, 0.9, 0.0, 1.0] // yellow = available
        } else {
            [0.3, 0.3, 0.0, 0.4] // dim = on cooldown
        };
        let center = Vec3::new(px, 0.0, -py); // render space
        let r = if i < 6 { 180.0_f32 } else { 90.0_f32 };
        circle_lines(v, center, r, 1, color, 8);
    }
}
