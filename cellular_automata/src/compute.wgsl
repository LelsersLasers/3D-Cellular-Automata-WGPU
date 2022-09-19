struct Cell {
    position: array<f32, 3>,
	hp: i32,
	neighbors: i32,
    lights: array<f32, 8>,
};

struct Wrapped {
    x: i32,
    padding0: i32,
    padding1: i32,
    padding2: i32,
}

struct Rules {
    survival: array<Wrapped, 27>,
    spawn: array<Wrapped, 27>,
    state: i32,
    cell_bounds: u32,
};


@group(0) @binding(0)
var<uniform> rules: Rules;
@group(0) @binding(1)
var<storage, read_write> cells: array<Cell>;


fn three_to_one(x: u32, y: u32, z: u32) -> u32 {
    return z + y * rules.cell_bounds + x * rules.cell_bounds * rules.cell_bounds;
}

@compute @workgroup_size(1)
fn count_neighbors(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // let one_idx = three_to_one(x, y, z);
    // self.cells[one_idx].neighbors = 0;
    // for offset in NEIGHBOR_OFFSETS.iter() {
    //     if valid_idx(x, y, z, *offset) {
    //         if self.cells[three_to_one(
    //             (x as i32 + offset.0) as u32,
    //             (y as i32 + offset.1) as u32,
    //             (z as i32 + offset.2) as u32,
    //         )]
    //         .get_alive()
    //         {
    //             self.cells[one_idx].neighbors += 1;
    //         }
    //     }
    // }
    let idx = three_to_one(global_id.x, global_id.y, global_id.z);
    cells[idx].neighbors = 0;
    for (var x: i32 = -1; x < 2; x++) {
        for (var y: i32 = -1; y < 2; y++) {
            for (var z: i32 = -1; z < 2; z++) {
                if i32(global_id.x) + x >= 0
                    && i32(global_id.x) + x < i32(rules.cell_bounds)
                    && i32(global_id.y) + y >= 0
                    && i32(global_id.y) + y < i32(rules.cell_bounds)
                    && i32(global_id.z) + z >= 0
                    && i32(global_id.z) + z < i32(rules.cell_bounds)
                    && cells[three_to_one(
                        u32(i32(global_id.x) + x),
                        u32(i32(global_id.y) + y),
                        u32(i32(global_id.z) + z)
                    )].hp == rules.state {
                    cells[idx].neighbors += 1;
                } else if x == 0 && y == 0 && z == 0 {
                    continue;
                }
            }
        }
    }
}


@compute @workgroup_size(1)
fn sync(@builtin(global_invocation_id) global_id: vec3<u32>) {
	// self.hp = (self.hp == STATE as i32) as i32 * (self.hp - 1 + SURVIVAL[self.neighbors as usize] as i32) + // alive
    //     (self.hp < 0) as i32 * (SPAWN[self.neighbors as usize] as i32 * (STATE + 1) as i32 - 1) +  // dead
    //     (self.hp >= 0 && self.hp < STATE as i32) as i32 * (self.hp - 1); // dying

    let idx = three_to_one(global_id.x, global_id.y, global_id.z);
    cells[idx].hp =
        i32(cells[idx].hp == rules.state) * (cells[idx].hp - 1 + rules.survival[cells[idx].neighbors].x) + // alive
        i32(cells[idx].hp < 0) * (rules.spawn[cells[idx].neighbors].x * (rules.state + 1) - 1) +  // dead
        i32(cells[idx].hp >= 0 && cells[idx].hp < rules.state) * (cells[idx].hp - 1); // dying
}