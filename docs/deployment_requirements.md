# Deployment requirements (pre-hosting assessment)

This is a requirements assessment, not a deployment plan. The current application is single-athlete and local-first. Known historical activities may legitimately have metadata without recoverable FIT/TCX detail; that alone does not block deployment.

## Runtime and persistent state

- Python 3.11 or newer, the packages in `pyproject.toml`, and Playwright Chromium installed for the same user that runs sync.
- A persistent writable SQLite database (`ENDURANCE_DB_PATH`, default `data/endurance_lab.sqlite3`). Preserve its WAL/SHM files during operation and use SQLite's backup API or a stopped service for backups.
- Persistent `data/raw/` source copies, `training_data/import/` exports, `training_data/manifest.csv`, private logs/reports, and `config/athlete.yaml`.
- Persistent, access-restricted `training_data/auth/` browser profile and Google Calendar OAuth client/token. The browser profile is a credential. Never bake these into an image or commit them.
- Optional path overrides: `ENDURANCE_DB_PATH`, `ENDURANCE_IMPORT_DIR`, `ENDURANCE_RAW_DIR`, `ENDURANCE_ATHLETE_CONFIG`. `ENDURANCE_BIND_ADDRESS` controls the local dashboard bind address. Dashboard default port: 8501.
- One scheduled sync worker. The file lock and SQLite single-writer model assume one shared filesystem, not independent replicas. The dashboard may run alongside the worker.
- Authentication and TLS in front of Streamlit. The present ngrok Basic Authentication policy protects local phone access; it is not a general hosted identity system.

## Platform suitability

| Platform | Current architecture | Constraint |
| --- | --- | --- |
| Single persistent VM or container | Feasible with engineering work | Attach durable private storage, install Chromium, run a scheduler and dashboard, provide authenticated HTTPS, backups, monitoring, and controlled restarts. |
| Stateless PaaS | Not safe as-is | Ephemeral disk loses SQLite, exports, browser profile, tokens, and lock state. Multiple replicas can race. |
| Streamlit-hosted environment | Not safe as-is | Dashboard-only hosting does not provide a durable authenticated browser profile, archive storage, scheduled sync, and private single-writer database together. |

## Windows assumptions to replace

The current-user Task Scheduler/Startup launchers and portable ngrok setup are Windows-specific. A hosted system needs its own process supervisor, scheduler, secret store, backup policy, browser installation, health checks, and authenticated ingress. Path handling and file-lock behavior must be tested on the target OS. Do not port the saved Windows browser profile blindly; re-authorize in the target environment.

## Release gate before hosting

The archive must distinguish total known activities from detailed source coverage and export questions. Old metadata-only workouts do not need to be re-exported before hosting. Real release gates are recent sync reliability, unexpected missing exports for normal device-recorded activities, valid authentication, no duplicate/corrupt activities, a verified backup/restore drill, database integrity, a clean test run, calendar idempotency, and a privacy audit. Official Strava OAuth/webhooks may replace website scraping in a later architecture phase, but are not implemented here.
