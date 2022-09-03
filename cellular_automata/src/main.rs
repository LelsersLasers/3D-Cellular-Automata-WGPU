use cellular_automata::run;

fn main() {
    println!("Starting...");

    pollster::block_on(run());

    println!("Exiting...");
}
