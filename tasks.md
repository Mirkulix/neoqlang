# Aufgabenplan – Repository-Hardening und Konsistenz

## 1) Governance (Root)
- Dateien erstellen: LICENSE, CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md, CHANGELOG.md.
- Root-README aktualisieren: Verweis auf Governance-Dateien, Monorepo-Übersicht.

## 2) CI/CD
- Rust-Workflow erweitern:
  - Matrix: ubuntu-latest, macos-latest, windows-latest.
  - Schritte: cargo build --workspace, cargo test --workspace, fmt, clippy (als Fehler).
  - Caching anpassen pro OS.
- Python-Workflow (IGQK & SaaS):
  - Install: pip install -r requirements.txt (+ dev requirements).
  - Lint: ruff check, black --check, mypy.
  - Tests: pytest mit coverage; Upload zu Codecov.
- Node/Editor:
  - Build Theia und VSCode in separatem Job.
- Security:
  - cargo-deny/audit, pip-audit, npm/yarn audit.
- Dependabot:
  - .github/dependabot.yml für cargo, pip, npm.

## 3) Docker/Compose
- qlang/docker-compose.yml:
  - command ["serve"] auf ["web"] ändern.
  - worker-Service temporär entfernen/kommentieren.
  - Images (z. B. nginx) auf Digest pinnen (SaaS compose).
- README anpassen: Start-/Stop-Befehle, Ports, ENV.

## 4) Secrets & ENV
- SaaS Backend:
  - python-dotenv beim Start laden.
  - JWT Secret und ähnliche Werte aus ENV beziehen; harte Konstanten entfernen.
  - .env.example pflegen.
- Dokumentation: ENV-Variablen beschreiben.

## 5) Security & Netzwerk
- Rust Ollama-Client:
  - Feature-Flag „tls“ definieren.
  - Implementationspfad skizzieren (rustls) oder Reverse-Proxy-Option beschreiben.
- SECURITY.md pflegen (Policy, Meldeweg, unterstützte Versionen).

## 6) Python Packaging & Tests
- IGQK:
  - pyproject.toml (PEP 621) hinzufügen (name, version, dependencies, optional-extras[dev]).
  - build-backend (setuptools/hatch) wählen.
  - tests/ vereinheitlichen; pytest-Konfiguration (pyproject oder pytest.ini).
  - Tooling-Konfig (ruff/black/mypy) in pyproject.
  - Redundante setup.py entfernen/vereinheitlichen.

## 7) Dokumentation & DX
- qlang/README:
  - Feature-Flags (llvm, gpu, mlx) dokumentieren.
  - Badges (docs.rs/crates.io), falls publiziert.
- SaaS/README:
  - Verweis auf /docs korrigieren; Gradio-UI klar benennen.
- Root-README:
  - Projektübersicht, Komponenten, Schnellstart, Build/Run/Test Leitfaden.

## 8) Observability
- SaaS:
  - prometheus-client integrieren; /metrics Endpoint.
  - strukturierte Logs (Python logging mit JSON-Formatter).
- Rust (optional):
  - tracing einführen (Basis-Subscriber), env-gesteuert.

## 9) Node/Editor
- Theia:
  - yarn.lock erzeugen und einchecken.
  - CI-Job für Build (yarn install --frozen-lockfile, yarn build).
- VSCode:
  - CI-Build validieren.

## 10) Follow-up (optional)
- Lizenz-Header-Check im CI.
- crates.io/docs.rs Release-Workflow für Rust-Crates (wenn gewünscht).

