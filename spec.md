# Repository-Hardening und Konsistenz – Spezifikation

## Hintergrund
Die Codeanalyse zeigt Lücken in Governance, CI/CD, Security, Packaging, Docker-Setup und Dokumentation über mehrere Teilprojekte (QLANG/Rust, IGQK/Python, SaaS/Backend, Editor-Integrationen). Diese Spezifikation definiert Ziele, Umfang und Anforderungen, um das Monorepo konsistent, sicherer und leichter wartbar zu machen.

## Ziele
- Einheitliche Projekt-Governance im Repo-Root.
- Korrekte, reproduzierbare Laufumgebungen (Docker/ENV) ohne harte Secrets.
- CI/CD deckt Rust, Python und Editor-Pakete ab; Security-Prüfungen und Coverage integriert.
- Python-Packaging modernisieren und Tests vereinheitlichen.
- Korrekte Docker-Subcommands für QLANG, robuste Dokumentation und Developer Experience.
- Basis-Observability für SaaS (Logging/Metrics).

## Nicht-Ziele
- Keine funktionalen Erweiterungen des QLANG-Compilers oder IGQK-Algorithmen.
- Keine tiefgreifende UI-Neuentwicklung.

## Umfang
- QLANG (Rust Workspace)
- IGQK (Python Framework) inkl. IGQK_Complete_Package
- IGQK SaaS Backend
- Editor-Integrationen (Theia, VSCode)
- Repo-Root (Governance/CI/Security)

## Anforderungen
1) Governance (Repo-Root)
- LICENSE (übergreifend, MIT vorgeschlagen; Konsistenz mit qlang/LICENSE).
- CONTRIBUTING.md (Issue/PR-Workflow, lokale Checks).
- CODE_OF_CONDUCT.md.
- SECURITY.md (Meldeweg, unterstützte Versionen).
- CHANGELOG.md (Keep a Changelog), konform mit Release-Prozess.

2) CI/CD
- Rust: Matrix (ubuntu-latest, macos-latest, windows-latest), build+test+clippy+fmt, Artefakte optional.
- Python (IGQK & SaaS): ruff/black/mypy/pytest (+pytest-cov), Coverage-Upload (Codecov).
- Node/Editor: Build-Test für VSCode-Extension und Theia.
- Security: cargo-deny oder cargo-audit; pip-audit; npm/yarn audit.
- Dependabot-Konfiguration für Cargo, Pip, npm/yarn.

3) Docker/Compose
- Korrektur qlang/docker-compose.yml: „serve“ → vorhandenes „web“ Subcommand; „worker“ entfernen oder deaktivieren, bis implementiert.
- Healthchecks beibehalten; Images (z. B. nginx) mit Digest pinnen.
- Dokumentation: Startbefehle und ENV-Variablen eindeutig beschreiben.

4) Secrets & ENV
- Keine harten Secrets im Code (JWT Secret). Nutzung von .env; Laden via python-dotenv im SaaS-Backend.
- Dokumentierte .env.example mit minimalen Pflichtvariablen.
- Optional: Secret-Management-Hinweise (GitHub Actions Secrets).

5) Security & Netzwerk
- Ollama-Client: Option für TLS/HTTPS. Ansatz: Feature-Flag „tls“ im Runtime-Crate; bei Aktivierung Verwendung von rustls-basiertem Client oder klar dokumentierte Reverse-Proxy-Variante.
- SECURITY.md enthält Support-Policy und Eskalationswege.

6) Python Packaging & Tests
- pyproject.toml (PEP 621) für IGQK, build-backend (setuptools oder hatch), Entfernen redundanter setup.py, Vereinheitlichung.
- Teststruktur: tests/ im Paket; zentrale pytest-Konfiguration (pyproject/pytest.ini).
- Tooling-Konfig: ruff/black/mypy in pyproject.toml (oder setup.cfg).

7) Dokumentation & DX
- qlang/README: Dokumentation der Cargo-Features (llvm, gpu, mlx), Badges (docs.rs/crates.io falls publiziert).
- SaaS/README: Korrektur des Verweises auf /docs; Klarstellung, dass UI Gradio ist.
- Root-README: Monorepo-Übersicht, Teilprojekte, Schnellstart.

8) Observability
- SaaS: /metrics Endpoint via prometheus-client; strukturierte Logs (Python logging mit JSON-Formatter).
- Rust: Baseline tracing (tracing crate) optional in Runtime aktiviert.

9) Node/Editor
- Theia: Lockfile (yarn.lock) einchecken; CI-Build.
- VSCode: Build in CI validieren.

## Abnahmekriterien
- Root-Governance-Dateien vorhanden, referenziert im Root-README.
- CI läuft grün für Rust, Python und Editor-Builds auf mindestens Linux; Matrix für Rust aktiv.
- Security-Checks laufen in CI; Dependabot aktiv.
- Keine harten Secrets im Code; dotenv wird im SaaS beim Start geladen; .env.example aktuell.
- qlang/docker-compose.yml nutzt existierende CLI-Kommandos; Services starten lokal erfolgreich.
- IGQK besitzt pyproject.toml; Tests laufen via pytest; Coverage-Report erzeugt und optional hochgeladen.
- qlang/README dokumentiert Feature-Flags; SaaS-README ist konsistent mit tatsächlicher Implementierung.
- SaaS bietet /metrics und erzeugt strukturierte Logs.

## Risiken/Entscheidungen
- TLS im Rust-Client: Optionales Feature-Flag begrenzt Abhängigkeiten; alternativ Dokumentation eines TLS-fähigen Reverse-Proxys.
- Windows/macOS Matrix kann Buildzeiten erhöhen; Cache-Strategie sicherstellen.

