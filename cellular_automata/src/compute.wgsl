struct Cells {
	hps: [i32; 884736],
	neighbors: [i32; 884736],
	state: i32,
	survival: array<bool>,
	spawn: array<bool>
}

@group(0) @binding(0)
var<storage, read_write> cells: Cells;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
	cells.hps[global_id] = (cells.hps[global_id] == cells.state) * (cells.hps[global_id] - 1 + cells.survival[cells.neighbors[global_id]]) + // alive
		(cells.hps[global_id] < 0) * (cells.spawn[cells.neighbors[global_id]] * (cells.state + 1) - 1) +  // dead
		(cells.hps[global_id] >= 0 && cells.hps[global_id] < cells.state) * (cells.hps[global_id] - 1);
}