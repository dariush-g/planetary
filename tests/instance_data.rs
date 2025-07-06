use glam::{Mat4, Vec3};
use planetary::state::InstanceData;

#[test]
fn test_instance_pad_align() {
    let instances = vec![
        InstanceData::new(
            Mat4::IDENTITY,
            1.0,
            [1.0, 0.0, 0.0], // RED
        ),
        InstanceData::new(
            Mat4::from_translation(Vec3::new(3.0, 0.0, 0.0)),
            1.0,
            [0.0, 0.0, 1.0], // BLUE
        ),
    ];

    println!("CPU Colors:");
    println!("- Red: {:?}", instances[0].color);
    println!("- Blue: {:?}", instances[1].color);

    assert!(std::mem::size_of::<InstanceData>() == 96);
    assert_eq!(std::mem::size_of_val(&instances[0]), 96);
}
