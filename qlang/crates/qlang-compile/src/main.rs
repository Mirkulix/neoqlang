//! QLANG CLI — Compile and execute QLANG graph files.
//!
//! Usage:
//!   qlang-cli run <file.qlg.json>    Execute a graph from JSON file
//!   qlang-cli info <file.qlg.json>   Show graph information
//!   qlang-cli verify <file.qlg.json> Verify graph constraints
//!   qlang-cli optimize <file.qlg.json> [-o output.qlg.json]  Optimize and save

use std::collections::HashMap;
use std::env;
use std::fs;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        print_usage();
        process::exit(1);
    }

    let command = &args[1];
    let file_path = &args[2];

    // Read the graph file
    let content = match fs::read_to_string(file_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error reading {file_path}: {e}");
            process::exit(1);
        }
    };

    let graph = match qlang_core::serial::from_json(&content) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Error parsing graph: {e}");
            process::exit(1);
        }
    };

    match command.as_str() {
        "info" => cmd_info(&graph),
        "verify" => cmd_verify(&graph),
        "optimize" => {
            let output = args.get(4).map(|s| s.as_str());
            cmd_optimize(graph, output);
        }
        "run" => cmd_run(&graph),
        _ => {
            eprintln!("Unknown command: {command}");
            print_usage();
            process::exit(1);
        }
    }
}

fn print_usage() {
    eprintln!("QLANG CLI v0.1");
    eprintln!();
    eprintln!("Usage:");
    eprintln!("  qlang-cli info     <file.qlg.json>                    Show graph info");
    eprintln!("  qlang-cli verify   <file.qlg.json>                    Verify constraints");
    eprintln!("  qlang-cli optimize <file.qlg.json> -o <output.json>   Optimize graph");
    eprintln!("  qlang-cli run      <file.qlg.json>                    Execute graph");
}

fn cmd_info(graph: &qlang_core::graph::Graph) {
    println!("{graph}");

    let inputs = graph.input_nodes();
    let outputs = graph.output_nodes();
    let quantum_ops: Vec<_> = graph.nodes.iter().filter(|n| n.op.is_quantum()).collect();

    println!("Summary:");
    println!("  Inputs:      {}", inputs.len());
    println!("  Outputs:     {}", outputs.len());
    println!("  Total nodes: {}", graph.nodes.len());
    println!("  Total edges: {}", graph.edges.len());
    println!("  Quantum ops: {}", quantum_ops.len());

    // Binary size estimate
    if let Ok(binary) = qlang_core::serial::to_binary(graph) {
        println!("  Binary size: {} bytes", binary.len());
    }
}

fn cmd_verify(graph: &qlang_core::graph::Graph) {
    let result = qlang_core::verify::verify_graph(graph);
    println!("{result}");

    if result.is_ok() {
        println!("Graph verification PASSED.");
    } else {
        println!("Graph verification FAILED.");
        process::exit(1);
    }
}

fn cmd_optimize(mut graph: qlang_core::graph::Graph, output: Option<&str>) {
    let before = graph.nodes.len();
    qlang_compile::optimize::optimize(&mut graph);
    let after = graph.nodes.len();

    println!("Optimization complete:");
    println!("  Nodes before: {before}");
    println!("  Nodes after:  {after}");
    println!("  Removed:      {}", before - after);

    if let Some(path) = output {
        let json = qlang_core::serial::to_json(&graph).unwrap();
        fs::write(path, json).unwrap();
        println!("  Saved to: {path}");
    }
}

fn cmd_run(graph: &qlang_core::graph::Graph) {
    // For CLI execution, create zero-filled inputs
    let mut inputs = HashMap::new();
    for node in graph.input_nodes() {
        if let qlang_core::ops::Op::Input { name } = &node.op {
            if let Some(tt) = node.output_types.first() {
                if let Some(data) = qlang_core::tensor::TensorData::zeros(tt) {
                    println!("  Input '{name}': {} (zeros)", tt);
                    inputs.insert(name.clone(), data);
                }
            }
        }
    }

    match qlang_runtime::executor::execute(graph, inputs) {
        Ok(result) => {
            println!("\nExecution complete:");
            println!("  Nodes executed: {}", result.stats.nodes_executed);
            println!("  Quantum ops:    {}", result.stats.quantum_ops);
            println!("  Total FLOPs:    {}", result.stats.total_flops);

            for (name, tensor) in &result.outputs {
                println!("\n  Output '{name}':");
                println!("    dtype: {}", tensor.dtype);
                println!("    shape: {}", tensor.shape);
                if let Some(vals) = tensor.as_f32_slice() {
                    if vals.len() <= 20 {
                        println!("    values: {:?}", vals);
                    } else {
                        println!("    values: [{}, {}, ... {} total]", vals[0], vals[1], vals.len());
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("Execution failed: {e}");
            process::exit(1);
        }
    }
}
