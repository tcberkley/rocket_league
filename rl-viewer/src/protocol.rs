use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub fn to_glam(&self) -> glam::Vec3 {
        // RL coord system: X=lateral, Y=longitudinal (goal), Z=vertical
        // We map directly; camera handles orientation
        glam::Vec3::new(self.x, self.z, -self.y)
    }
}

/// 3x3 rotation matrix as row-major [[f32;3];3]
pub type RotMat = [[f32; 3]; 3];

pub fn rotmat_to_glam(r: &RotMat) -> glam::Mat3 {
    // r[row][col], columns are forward/right/up in RL space
    // Map to our render space: x=x, y=z, z=-y
    let col0 = glam::Vec3::new(r[0][0], r[2][0], -r[1][0]); // forward
    let col1 = glam::Vec3::new(r[0][1], r[2][1], -r[1][1]); // right
    let col2 = glam::Vec3::new(r[0][2], r[2][2], -r[1][2]); // up
    glam::Mat3::from_cols(col0, col1, col2)
}

#[derive(Debug, Deserialize, Clone)]
pub struct BallData {
    pub pos: Vec3,
    pub vel: Vec3,
    pub rot: RotMat,
}

#[derive(Debug, Deserialize, Clone)]
pub struct CarData {
    pub pos: Vec3,
    pub vel: Vec3,
    pub rot: RotMat,
    pub boost: f32,
    pub team: u8,
    pub on_ground: bool,
}

#[derive(Debug, Deserialize, Clone)]
pub struct FrameData {
    pub tick: u64,
    pub ball: BallData,
    pub cars: Vec<CarData>,
    pub boost_pads: Vec<f32>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Message {
    Frame(FrameData),
    Reset,
    Close,
}
