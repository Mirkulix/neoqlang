# Changelog
All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning where applicable.

## [Unreleased]
### Added
- Root governance files: LICENSE, CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, CHANGELOG.
- QLANG: Docker Compose uses `web` subcommand; removed unsupported `worker`.
- QLANG CLI: `serve` alias for `web`; tracing baseline (RUST_LOG).
- CI: OS matrix (no-LLVM), cargo-audit security job, coverage via tarpaulin.
- Dependabot for cargo/pip/npm.
- Runtime: Optional `tls` feature for Ollama client (rustls).

### Changed
- qlang/README: Documented cargo features and networking/HTTPS guidance.

### Fixed
- Compose mismatch with non-existent CLI subcommands.
