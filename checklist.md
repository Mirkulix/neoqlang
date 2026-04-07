# Checkliste – Repository-Hardening

- [ ] LICENSE im Repo-Root hinzugefügt und mit qlang/LICENSE konsistent
- [ ] CONTRIBUTING.md erstellt (PR-Workflow, lokale Checks)
- [ ] CODE_OF_CONDUCT.md erstellt
- [ ] SECURITY.md erstellt (Meldeweg, unterstützte Versionen)
- [ ] CHANGELOG.md begonnen (Keep a Changelog)
- [ ] Root-README mit Monorepo-Übersicht und Governance-Verweisen aktualisiert

- [ ] Rust CI: OS-Matrix aktiviert (Linux/macOS/Windows)
- [ ] Rust CI: build/test/fmt/clippy laufen grün
- [x] Python CI: ruff/black/mypy/pytest+coverage eingerichtet
- [x] Node CI: Theia & VSCode Build validiert
- [x] Security CI: cargo-deny/audit, pip-audit, npm/yarn audit integriert
- [ ] Dependabot aktiviert (.github/dependabot.yml)
- [ ] Coverage-Upload (Codecov) aktiv

- [ ] qlang/docker-compose: „serve“ → „web“ korrigiert
- [ ] qlang/docker-compose: „worker“-Service entfernt/deaktiviert bis implementiert
- [ ] SaaS compose: Images mit Digest gepinnt
- [ ] Compose-Dokumentation aktualisiert (Start/Ports/ENV)

- [ ] SaaS lädt .env via python-dotenv
- [ ] Harte Secrets entfernt (JWT etc. aus ENV)
- [ ] .env.example gepflegt (Pflichtvariablen dokumentiert)

- [ ] Ollama-Client: TLS-Option spezifiziert (Feature-Flag oder Reverse-Proxy beschrieben)
- [ ] SECURITY.md mit Support-Policy abgestimmt

- [x] IGQK: pyproject.toml hinzugefügt (PEP 621)
- [x] IGQK: Tests vereinheitlicht (tests/), pytest-Konfig zentral
- [x] IGQK: ruff/black/mypy in pyproject konfiguriert
- [x] IGQK: setup.py bereinigt/ersetzt

- [ ] qlang/README: Feature-Flags dokumentiert; Badges ergänzt
- [ ] SaaS/README: /docs-Verweis korrigiert; Gradio-UI benannt
- [ ] Root-README: Schnellstart und Komponentenübersicht ergänzt

- [ ] SaaS: /metrics Endpoint verfügbar
- [ ] SaaS: strukturierte Logs aktiv
- [ ] Rust: optionales tracing konfigurierbar

- [ ] Theia: yarn.lock eingecheckt; Build mit --frozen-lockfile
- [ ] VSCode: CI-Build OK

- [ ] Optional: Lizenz-Header-Check im CI
- [ ] Optional: crates.io/docs.rs Release-Workflow spezifiziert

