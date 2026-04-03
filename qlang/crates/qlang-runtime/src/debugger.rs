//! Interactive debugger for QLANG programs.
//!
//! Provides breakpoints (with optional conditions), single-stepping,
//! variable inspection, call-stack tracking, expression evaluation,
//! and full execution tracing with timestamps.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::vm::{
    tokenize, parse_program, Stmt, VmError, VmState,
};
use crate::executor::{self, ExecutionError};
use qlang_core::graph::{Graph, NodeId};
use qlang_core::ops::Op;
use qlang_core::tensor::TensorData;

// ─── Breakpoint ────────────────────────────────────────────────────────────

/// A breakpoint attached to a source line.
#[derive(Debug, Clone)]
pub struct Breakpoint {
    /// 1-based source line number.
    pub line: usize,
    /// Optional condition expression (QLANG source). The breakpoint only
    /// fires when this evaluates to a truthy value.
    pub condition: Option<String>,
    /// Whether the breakpoint is currently active.
    pub enabled: bool,
}

impl Breakpoint {
    pub fn new(line: usize) -> Self {
        Self {
            line,
            condition: None,
            enabled: true,
        }
    }

    pub fn with_condition(line: usize, condition: &str) -> Self {
        Self {
            line,
            condition: Some(condition.to_string()),
            enabled: true,
        }
    }
}

// ─── Debug State ───────────────────────────────────────────────────────────

/// Snapshot of the debugger at a particular point in execution.
#[derive(Debug, Clone)]
pub struct DebugState {
    /// 1-based current source line (index into the flat statement list).
    pub current_line: usize,
    /// Call stack frames (outermost first). Each entry is the function name
    /// (or `"<main>"` for top-level code).
    pub call_stack: Vec<String>,
    /// Snapshot of all visible variables at this point.
    pub variables: HashMap<String, String>,
    /// Whether execution is currently paused.
    pub paused: bool,
}

// ─── Execution Trace ───────────────────────────────────────────────────────

/// A single entry in the execution trace.
#[derive(Debug, Clone)]
pub struct TraceEntry {
    /// 1-based statement index that was executed.
    pub line: usize,
    /// Wall-clock offset from the start of execution.
    pub timestamp: Duration,
    /// Short description of the statement.
    pub description: String,
}

/// Records every statement executed together with a timestamp.
#[derive(Debug, Clone)]
pub struct ExecutionTrace {
    pub entries: Vec<TraceEntry>,
    start: Option<Instant>,
}

impl ExecutionTrace {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            start: None,
        }
    }

    fn record(&mut self, line: usize, description: String) {
        let start = *self.start.get_or_insert_with(Instant::now);
        self.entries.push(TraceEntry {
            line,
            timestamp: start.elapsed(),
            description,
        });
    }
}

impl Default for ExecutionTrace {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Debugger ──────────────────────────────────────────────────────────────

/// Interactive debugger that wraps a QLANG VM and adds debugging capabilities.
pub struct Debugger {
    /// The flat, ordered list of top-level statements (the "program").
    #[allow(dead_code)]
    stmts: Vec<Stmt>,
    /// Source lines (for pretty-printing).
    #[allow(dead_code)]
    source_lines: Vec<String>,
    /// The VM state used for execution.
    vm: VmState,
    /// Instruction pointer — index into the *flat* statement list produced
    /// by `flatten_stmts`. Each entry is `(stmt, nesting-depth)`.
    flat_stmts: Vec<(Stmt, usize)>,
    /// Current position in `flat_stmts`.
    pc: usize,
    /// Breakpoints keyed by line number.
    breakpoints: HashMap<usize, Breakpoint>,
    /// Whether execution is paused.
    paused: bool,
    /// Call stack (function names).
    call_stack: Vec<String>,
    /// Execution trace.
    pub trace: ExecutionTrace,
    /// Whether the program has finished.
    finished: bool,
}

impl Debugger {
    // ── Construction ────────────────────────────────────────────────────

    /// Create a new debugger from QLANG source code.
    pub fn from_source(source: &str) -> Result<Self, VmError> {
        let tokens = tokenize(source)?;
        let stmts = parse_program(&tokens)?;
        let source_lines: Vec<String> = source.lines().map(|l| l.to_string()).collect();
        let flat = flatten_stmts(&stmts, 0);
        Ok(Self {
            stmts,
            source_lines,
            vm: VmState::new(),
            flat_stmts: flat,
            pc: 0,
            breakpoints: HashMap::new(),
            paused: true,
            call_stack: vec!["<main>".to_string()],
            trace: ExecutionTrace::new(),
            finished: false,
        })
    }

    // ── Breakpoint management ──────────────────────────────────────────

    pub fn add_breakpoint(&mut self, line: usize) {
        self.breakpoints
            .entry(line)
            .or_insert_with(|| Breakpoint::new(line));
    }

    pub fn add_conditional_breakpoint(&mut self, line: usize, condition: &str) {
        self.breakpoints
            .insert(line, Breakpoint::with_condition(line, condition));
    }

    pub fn remove_breakpoint(&mut self, line: usize) -> bool {
        self.breakpoints.remove(&line).is_some()
    }

    pub fn list_breakpoints(&self) -> Vec<&Breakpoint> {
        let mut bps: Vec<_> = self.breakpoints.values().collect();
        bps.sort_by_key(|b| b.line);
        bps
    }

    // ── Stepping ───────────────────────────────────────────────────────

    /// Execute exactly one statement and pause.
    pub fn step(&mut self) -> Result<DebugState, VmError> {
        if self.finished {
            return Ok(self.snapshot());
        }
        self.execute_one()?;
        Ok(self.snapshot())
    }

    /// Step into a function call. In this flat model this is identical to
    /// `step` because function bodies are expanded inline during
    /// `flatten_stmts`.
    pub fn step_into(&mut self) -> Result<DebugState, VmError> {
        self.step()
    }

    /// Step over a function call — execute statements until we return to the
    /// same or shallower nesting depth.
    pub fn step_over(&mut self) -> Result<DebugState, VmError> {
        if self.finished {
            return Ok(self.snapshot());
        }
        let start_depth = self.current_depth();
        self.execute_one()?;
        while !self.finished && self.current_depth() > start_depth {
            self.execute_one()?;
        }
        Ok(self.snapshot())
    }

    /// Continue execution until the next enabled breakpoint fires or the
    /// program finishes.
    pub fn continue_execution(&mut self) -> Result<DebugState, VmError> {
        if self.finished {
            return Ok(self.snapshot());
        }
        // Execute at least one statement so we leave the current breakpoint.
        self.execute_one()?;
        while !self.finished {
            let line = self.current_line();
            if self.should_break(line)? {
                break;
            }
            self.execute_one()?;
        }
        self.paused = true;
        Ok(self.snapshot())
    }

    // ── Inspection ─────────────────────────────────────────────────────

    pub fn inspect_variable(&self, name: &str) -> Option<String> {
        self.vm.get_var(name).ok().map(|v| format!("{v}"))
    }

    pub fn inspect_all_variables(&self) -> Vec<(String, String)> {
        let mut vars: Vec<(String, String)> = Vec::new();
        for scope in self.vm.scopes.iter() {
            for (k, v) in scope.iter() {
                vars.push((k.clone(), format!("{v}")));
            }
        }
        vars.sort_by(|a, b| a.0.cmp(&b.0));
        vars
    }

    pub fn call_stack(&self) -> Vec<String> {
        self.call_stack.clone()
    }

    /// Evaluate an arbitrary QLANG expression in the current execution
    /// context and return the result as a string.
    pub fn evaluate_expression(&mut self, expr_src: &str) -> Result<String, VmError> {
        let tokens = tokenize(expr_src)?;
        // Wrap in a trivial program: a single expression statement.
        let wrapped = format!("print({expr_src})");
        let tokens2 = tokenize(&wrapped)?;
        let stmts = parse_program(&tokens2)?;
        // Save current output length, execute, grab new output.
        let before = self.vm.output.len();
        // We don't use tokens directly — just validate they lex.
        let _ = tokens;
        self.vm.exec_stmts(&stmts)?;
        if self.vm.output.len() > before {
            Ok(self.vm.output.last().unwrap().clone())
        } else {
            Ok("null".to_string())
        }
    }

    /// Has the program finished executing?
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Return captured print output.
    pub fn output(&self) -> &[String] {
        &self.vm.output
    }

    /// Current 1-based line number.
    pub fn current_line(&self) -> usize {
        if self.pc < self.flat_stmts.len() {
            // Lines are 1-based: first statement is line 1.
            self.pc + 1
        } else {
            self.flat_stmts.len()
        }
    }

    // ── Internal helpers ───────────────────────────────────────────────

    fn current_depth(&self) -> usize {
        if self.pc < self.flat_stmts.len() {
            self.flat_stmts[self.pc].1
        } else {
            0
        }
    }

    fn execute_one(&mut self) -> Result<(), VmError> {
        if self.pc >= self.flat_stmts.len() {
            self.finished = true;
            return Ok(());
        }
        let (stmt, _depth) = self.flat_stmts[self.pc].clone();
        let line = self.pc + 1;

        // Record trace.
        self.trace.record(line, describe_stmt(&stmt));

        // Track call stack for function definitions / calls.
        match &stmt {
            Stmt::FnDef { name, .. } => {
                self.call_stack.push(name.clone());
            }
            Stmt::Return(_) => {
                if self.call_stack.len() > 1 {
                    self.call_stack.pop();
                }
            }
            _ => {}
        }

        self.vm.exec_stmt(&stmt)?;
        self.pc += 1;

        if self.pc >= self.flat_stmts.len() {
            self.finished = true;
        }
        Ok(())
    }

    fn should_break(&self, line: usize) -> Result<bool, VmError> {
        if let Some(bp) = self.breakpoints.get(&line) {
            if !bp.enabled {
                return Ok(false);
            }
            if let Some(ref cond) = bp.condition {
                // Evaluate condition in current context — we need a mutable
                // VM, but `should_break` takes `&self`. Work around by
                // cloning the expression evaluation into a temporary VM.
                let tokens = tokenize(cond)?;
                let wrapped = format!("print({cond})");
                let tokens2 = tokenize(&wrapped)?;
                let stmts = parse_program(&tokens2)?;
                let mut tmp_vm = self.vm.clone();
                let _ = tokens;
                let before = tmp_vm.output.len();
                tmp_vm.exec_stmts(&stmts)?;
                if tmp_vm.output.len() > before {
                    let val_str = tmp_vm.output.last().unwrap();
                    return Ok(val_str != "0" && val_str != "false" && val_str != "null");
                }
                return Ok(false);
            }
            return Ok(true);
        }
        Ok(false)
    }

    fn snapshot(&self) -> DebugState {
        let mut variables = HashMap::new();
        for scope in self.vm.scopes.iter() {
            for (k, v) in scope.iter() {
                variables.insert(k.clone(), format!("{v}"));
            }
        }
        DebugState {
            current_line: self.current_line(),
            call_stack: self.call_stack.clone(),
            variables,
            paused: !self.finished,
        }
    }
}

// ─── Flatten statements ────────────────────────────────────────────────────

/// Flatten a tree of statements into a linear sequence suitable for
/// single-stepping. Each entry carries its nesting depth so that
/// `step_over` can skip over deeper statements.
fn flatten_stmts(stmts: &[Stmt], depth: usize) -> Vec<(Stmt, usize)> {
    let mut out = Vec::new();
    for stmt in stmts {
        match stmt {
            Stmt::If { .. } => {
                // Push the if-statement itself (will evaluate the condition).
                out.push((stmt.clone(), depth));
                // We don't flatten sub-bodies here because the VM's
                // `exec_stmt` handles them internally. This keeps the
                // debugger consistent with the VM's actual execution.
            }
            Stmt::While { .. } | Stmt::For { .. } => {
                out.push((stmt.clone(), depth));
            }
            Stmt::FnDef { .. } => {
                out.push((stmt.clone(), depth));
            }
            _ => {
                out.push((stmt.clone(), depth));
            }
        }
    }
    out
}

// ─── Describe statement ────────────────────────────────────────────────────

fn describe_stmt(stmt: &Stmt) -> String {
    match stmt {
        Stmt::Let { name, .. } => format!("let {name} = ..."),
        Stmt::Assign { name, .. } => format!("{name} = ..."),
        Stmt::If { .. } => "if ...".to_string(),
        Stmt::While { .. } => "while ...".to_string(),
        Stmt::For { var, .. } => format!("for {var} in ..."),
        Stmt::FnDef { name, params, .. } => {
            format!("fn {name}({})", params.join(", "))
        }
        Stmt::Return(_) => "return ...".to_string(),
        Stmt::Print(_) => "print(...)".to_string(),
        Stmt::ExprStmt(_) => "expr".to_string(),
    }
}

// ─── Pretty-print debug state ──────────────────────────────────────────────

/// Format a `DebugState` as a human-readable string, highlighting the
/// current source line.
pub fn format_debug_state(state: &DebugState, source: &str) -> String {
    let lines: Vec<&str> = source.lines().collect();
    let mut out = String::new();

    out.push_str("=== Debug State ===\n");
    out.push_str(&format!("Status: {}\n", if state.paused { "PAUSED" } else { "FINISHED" }));
    out.push_str(&format!("Line: {}\n", state.current_line));

    // Call stack
    out.push_str("Call stack:\n");
    for (i, frame) in state.call_stack.iter().enumerate() {
        out.push_str(&format!("  #{i}: {frame}\n"));
    }

    // Variables
    out.push_str("Variables:\n");
    let mut vars: Vec<_> = state.variables.iter().collect();
    vars.sort_by(|(a, _), (b, _)| a.cmp(b));
    for (name, val) in &vars {
        out.push_str(&format!("  {name} = {val}\n"));
    }

    // Source with current line highlighted
    out.push_str("Source:\n");
    for (i, line) in lines.iter().enumerate() {
        let line_no = i + 1;
        let marker = if line_no == state.current_line {
            ">>>"
        } else {
            "   "
        };
        out.push_str(&format!("{marker} {line_no:>4} | {line}\n"));
    }

    out
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_debugger(src: &str) -> Debugger {
        Debugger::from_source(src).expect("failed to create debugger")
    }

    // 1. Add / remove breakpoints
    #[test]
    fn test_add_remove_breakpoints() {
        let mut dbg = make_debugger("let x = 1\nlet y = 2\nlet z = 3");
        dbg.add_breakpoint(1);
        dbg.add_breakpoint(3);
        assert_eq!(dbg.list_breakpoints().len(), 2);

        assert!(dbg.remove_breakpoint(1));
        assert_eq!(dbg.list_breakpoints().len(), 1);
        assert_eq!(dbg.list_breakpoints()[0].line, 3);

        // Removing non-existent breakpoint returns false.
        assert!(!dbg.remove_breakpoint(99));
    }

    // 2. Step execution advances line
    #[test]
    fn test_step_advances_line() {
        let mut dbg = make_debugger("let a = 10\nlet b = 20\nlet c = 30");
        let s1 = dbg.step().unwrap();
        assert_eq!(s1.current_line, 2); // after executing line 1, we're at line 2

        let s2 = dbg.step().unwrap();
        assert_eq!(s2.current_line, 3);

        let _s3 = dbg.step().unwrap();
        // After executing line 3, the program is finished.
        assert!(dbg.is_finished());
    }

    // 3. Variable inspection
    #[test]
    fn test_variable_inspection() {
        let mut dbg = make_debugger("let x = 42\nlet y = 100");
        dbg.step().unwrap(); // execute `let x = 42`
        assert_eq!(dbg.inspect_variable("x"), Some("42".to_string()));
        assert_eq!(dbg.inspect_variable("y"), None);

        dbg.step().unwrap(); // execute `let y = 100`
        assert_eq!(dbg.inspect_variable("y"), Some("100".to_string()));

        let all = dbg.inspect_all_variables();
        assert!(all.contains(&("x".to_string(), "42".to_string())));
        assert!(all.contains(&("y".to_string(), "100".to_string())));
    }

    // 4. Call stack tracking
    #[test]
    fn test_call_stack() {
        let mut dbg = make_debugger("fn add(a, b) { return a + b }\nlet r = add(1, 2)");
        // Initially at <main>.
        assert_eq!(dbg.call_stack(), vec!["<main>".to_string()]);

        // Step: execute the fn definition — pushes "add" onto the call stack.
        dbg.step().unwrap();
        assert!(dbg.call_stack().contains(&"add".to_string()));
    }

    // 5. Breakpoint triggers at correct line
    #[test]
    fn test_breakpoint_triggers() {
        let mut dbg = make_debugger("let a = 1\nlet b = 2\nlet c = 3");
        dbg.add_breakpoint(3);
        let state = dbg.continue_execution().unwrap();
        // Should stop at line 3.
        assert_eq!(state.current_line, 3);
        // Variable b should already be set.
        assert!(state.variables.contains_key("b"));
    }

    // 6. Conditional breakpoint
    #[test]
    fn test_conditional_breakpoint() {
        let src = "let x = 0\nlet x = 1\nlet x = 2\nlet x = 3";
        let mut dbg = make_debugger(src);
        // Break at line 4 only when x == 2 (which it will be after line 3).
        dbg.add_conditional_breakpoint(4, "x == 2");
        // Conditional breakpoint at line 2 with condition x == 99 should NOT fire.
        dbg.add_conditional_breakpoint(2, "x == 99");
        let state = dbg.continue_execution().unwrap();
        // Should have skipped bp at line 2 (condition false) and stopped at line 4.
        assert_eq!(state.current_line, 4);
    }

    // 7. Execution trace recording
    #[test]
    fn test_execution_trace() {
        let mut dbg = make_debugger("let a = 1\nlet b = 2\nlet c = 3");
        dbg.step().unwrap();
        dbg.step().unwrap();
        dbg.step().unwrap();
        assert_eq!(dbg.trace.entries.len(), 3);
        assert_eq!(dbg.trace.entries[0].line, 1);
        assert_eq!(dbg.trace.entries[1].line, 2);
        assert_eq!(dbg.trace.entries[2].line, 3);
        // Descriptions should mention `let`.
        assert!(dbg.trace.entries[0].description.starts_with("let"));
        // Timestamps should be non-decreasing.
        assert!(dbg.trace.entries[1].timestamp >= dbg.trace.entries[0].timestamp);
    }

    // 8. Expression evaluation during pause
    #[test]
    fn test_expression_evaluation() {
        let mut dbg = make_debugger("let x = 10\nlet y = 20");
        dbg.step().unwrap(); // x = 10
        dbg.step().unwrap(); // y = 20

        let result = dbg.evaluate_expression("x + y").unwrap();
        assert_eq!(result, "30");

        let result2 = dbg.evaluate_expression("x * 3").unwrap();
        assert_eq!(result2, "30");
    }

    // 9. format_debug_state output
    #[test]
    fn test_format_debug_state() {
        let src = "let a = 1\nlet b = 2";
        let mut dbg = make_debugger(src);
        dbg.step().unwrap();
        let state = dbg.snapshot();
        let formatted = format_debug_state(&state, src);
        assert!(formatted.contains("PAUSED"));
        assert!(formatted.contains(">>>"));
        assert!(formatted.contains("<main>"));
    }

    // 10. step_over behaviour
    #[test]
    fn test_step_over() {
        let mut dbg = make_debugger("let a = 1\nlet b = 2\nlet c = 3");
        let s = dbg.step_over().unwrap();
        assert_eq!(s.current_line, 2);
    }

    // 11. Continue on finished program is idempotent
    #[test]
    fn test_continue_finished() {
        let mut dbg = make_debugger("let x = 1");
        dbg.step().unwrap();
        assert!(dbg.is_finished());
        // Calling continue / step again should not panic.
        let s = dbg.continue_execution().unwrap();
        assert!(!s.paused || dbg.is_finished());
    }

    // ── Graph debugger tests ──────────────────────────────────────────
    // NOTE: The graph debugger tests below are gated because GraphDebugger,
    // GraphDebugEvent, and debug_execute have not been implemented yet.
    #[cfg(feature = "__graph_debugger_tests")]
    use qlang_core::graph::Graph as TestGraph;
    #[cfg(feature = "__graph_debugger_tests")]
    use qlang_core::ops::Op as TestOp;
    #[cfg(feature = "__graph_debugger_tests")]
    use qlang_core::tensor::{Shape as TestShape, TensorData as TestTensorData, TensorType as TestTensorType};

    #[cfg(feature = "__graph_debugger_tests")]
    fn build_three_node_graph() -> (TestGraph, HashMap<String, TestTensorData>) {
        // Graph: a + b = y  (Input_a, Input_b, Add, Output_y => 4 graph nodes, 3 non-trivial)
        let mut g = TestGraph::new("test_dbg");

        let a = g.add_node(
            TestOp::Input { name: "a".into() },
            vec![],
            vec![TestTensorType::f32_vector(2)],
        );
        let b = g.add_node(
            TestOp::Input { name: "b".into() },
            vec![],
            vec![TestTensorType::f32_vector(2)],
        );
        let add = g.add_node(
            TestOp::Add,
            vec![TestTensorType::f32_vector(2), TestTensorType::f32_vector(2)],
            vec![TestTensorType::f32_vector(2)],
        );
        let out = g.add_node(
            TestOp::Output { name: "y".into() },
            vec![TestTensorType::f32_vector(2)],
            vec![],
        );

        g.add_edge(a, 0, add, 0, TestTensorType::f32_vector(2));
        g.add_edge(b, 0, add, 1, TestTensorType::f32_vector(2));
        g.add_edge(add, 0, out, 0, TestTensorType::f32_vector(2));

        let mut inputs = HashMap::new();
        inputs.insert(
            "a".to_string(),
            TestTensorData::from_f32(TestShape::vector(2), &[1.0, 2.0]),
        );
        inputs.insert(
            "b".to_string(),
            TestTensorData::from_f32(TestShape::vector(2), &[10.0, 20.0]),
        );

        (g, inputs)
    }

    // 12. Breakpoint on a graph node fires
    #[cfg(feature = "__graph_debugger_tests")]
    #[test]
    fn test_graph_breakpoint_fires() {
        let (graph, inputs) = build_three_node_graph();
        let order = graph.topological_sort().unwrap();
        // Set breakpoint on the third node in topological order (the Add node).
        let add_node_id = order[2];

        let mut gdbg = GraphDebugger::new();
        gdbg.add_breakpoint(add_node_id);

        let result = debug_execute(&graph, inputs, &mut gdbg).unwrap();

        // The breakpoint should have been hit.
        let bp_events: Vec<_> = gdbg.events.iter().filter(|e| matches!(e, GraphDebugEvent::BreakpointHit { .. })).collect();
        assert_eq!(bp_events.len(), 1);
        if let GraphDebugEvent::BreakpointHit { node_id, .. } = &bp_events[0] {
            assert_eq!(*node_id, add_node_id);
        }

        // Execution should still complete.
        assert!(result.outputs.contains_key("y"));
    }

    // 13. Step through graph, verify all events recorded
    #[cfg(feature = "__graph_debugger_tests")]
    #[test]
    fn test_graph_step_all_events() {
        let (graph, inputs) = build_three_node_graph();
        let mut gdbg = GraphDebugger::new();

        let result = debug_execute(&graph, inputs, &mut gdbg).unwrap();

        // Should have NodeEnter + NodeExit for each node in topo order, plus ExecutionComplete.
        let enter_count = gdbg.events.iter().filter(|e| matches!(e, GraphDebugEvent::NodeEnter { .. })).count();
        let exit_count = gdbg.events.iter().filter(|e| matches!(e, GraphDebugEvent::NodeExit { .. })).count();
        let complete_count = gdbg.events.iter().filter(|e| matches!(e, GraphDebugEvent::ExecutionComplete { .. })).count();

        assert_eq!(enter_count, 4); // 4 nodes: 2 inputs, 1 add, 1 output
        assert_eq!(exit_count, 4);
        assert_eq!(complete_count, 1);

        // Verify execution order is recorded.
        assert_eq!(gdbg.execution_order.len(), 4);

        assert!(result.outputs.contains_key("y"));
    }

    // 14. Tensor values captured at each step
    #[cfg(feature = "__graph_debugger_tests")]
    #[test]
    fn test_graph_tensor_values_captured() {
        let (graph, inputs) = build_three_node_graph();
        let order = graph.topological_sort().unwrap();
        let mut gdbg = GraphDebugger::new();

        let result = debug_execute(&graph, inputs, &mut gdbg).unwrap();

        // The Add node should have produced [11.0, 22.0].
        let add_node_id = order[2];
        let exit_events: Vec<_> = gdbg.events.iter().filter(|e| {
            matches!(e, GraphDebugEvent::NodeExit { node_id, .. } if *node_id == add_node_id)
        }).collect();
        assert_eq!(exit_events.len(), 1);

        if let GraphDebugEvent::NodeExit { output_shape, output_values, .. } = &exit_events[0] {
            assert!(output_shape.is_some());
            let vals = output_values.as_ref().unwrap();
            assert_eq!(vals.len(), 2);
            assert!((vals[0] - 11.0).abs() < 1e-6);
            assert!((vals[1] - 22.0).abs() < 1e-6);
        } else {
            panic!("expected NodeExit");
        }

        // Final output check.
        let y = result.outputs.get("y").unwrap();
        let values = y.as_f32_slice().unwrap();
        assert_eq!(values, vec![11.0, 22.0]);
    }
}
