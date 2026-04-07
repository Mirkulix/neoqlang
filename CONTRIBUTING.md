# Contributing

Thanks for your interest in contributing to QLANG!

## Getting Started
- Fork the repository and create a feature branch.
- Install Rust (stable), and for LLVM features install LLVM 18 (optional).
- Build workspace: `cd qlang && cargo build --workspace`
- Run tests: `cargo test --workspace`

## Coding Standards
- Rust: `cargo fmt --all` and `cargo clippy -- -D warnings`
- Keep public APIs documented. Prefer small, focused PRs.
- Do not commit secrets or tokens. Use `.env` locally only.

## Pull Requests
- Draft PRs welcome early for feedback.
- Include context and motivation in the PR description.
- Ensure CI passes (tests, lint, security audit).

## Security
- See SECURITY.md for how to report vulnerabilities.

## License
By contributing, you agree that your contributions are licensed under the MIT License (see LICENSE).
