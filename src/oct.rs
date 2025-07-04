use glam::Vec3;

#[derive(Clone, Debug)]
pub struct OctreeNode {
    center: Vec3,
    half_size: f32,
    mass: f32,
    center_of_mass: Vec3,
    children: [Option<Box<OctreeNode>>; 8],
    body: Option<Body>,
}

#[derive(Clone, Debug, Copy)]
struct Body {
    mass: f32,
    position: Vec3,
    velocity: Vec3,
    acceleration: Vec3,
}

impl OctreeNode {
    pub fn new(center: Vec3, half_size: f32) -> Self {
        let center_of_mass = center + Vec3::ZERO;

        Self {
            center,
            half_size,
            mass: 0.0,
            center_of_mass,
            children: [const { None }; 8],
            body: None,
        }
    }

    pub fn contains(&self, point: Vec3) -> bool {
        let min = self.center - Vec3::splat(self.half_size);
        let max = self.center + Vec3::splat(self.half_size);
        (min.x <= point.x)
            && (point.x <= max.x)
            && (min.y <= point.y)
            && (point.y <= max.y)
            && (min.z <= point.z)
            && (point.z <= max.z)
    }

    pub fn octant(&self, point: Vec3) -> usize {
        let mut index = 0;
        if point.x >= self.center.x {
            index |= 1;
        }
        if point.y >= self.center.y {
            index |= 2;
        }
        if point.z >= self.center.z {
            index |= 4;
        }
        index
    }
}
