struct Cell {
	hp: i32,
	neighbors: i32,
};
struct Cells {
    data: array<Cell>,
};

struct Wrapped {
    @size(16) x: i32
}

struct Rules {
    survival: array<Wrapped, 27>,
    spawn: array<Wrapped, 27>,
    state: i32,
};


@group(0) @binding(0)
var<uniform> rules: Rules;
@group(0) @binding(1)
var<storage, read_write> cells: Cells;


@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
	// self.hp = (self.hp == STATE as i32) as i32 * (self.hp - 1 + SURVIVAL[self.neighbors as usize] as i32) + // alive
    //     (self.hp < 0) as i32 * (SPAWN[self.neighbors as usize] as i32 * (STATE + 1) as i32 - 1) +  // dead
    //     (self.hp >= 0 && self.hp < STATE as i32) as i32 * (self.hp - 1); // dying
    let idx = global_id.z + global_id.y * u32(96) + global_id.x * u32(96) * u32(96);
    cells.data[idx].hp =
        i32(cells.data[idx].hp == rules.state) * (cells.data[idx].hp - 1 + rules.survival[cells.data[idx].neighbors].x) + // alive
        i32(cells.data[idx].hp < 0) * (rules.spawn[cells.data[idx].neighbors].x * (rules.state + 1) - 1) +  // dead
        i32(cells.data[idx].hp >= 0 && cells.data[idx].hp < rules.state) * (cells.data[idx].hp - 1); // dying
}