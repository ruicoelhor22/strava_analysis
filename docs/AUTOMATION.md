# Endurance Lab automation

Endurance Lab can run unattended under the current Windows user. It does not require an administrator account.

## What is installed

- `Endurance Lab Sync` is a Windows Task Scheduler task that runs the normal sync every 60 minutes while the user is logged in.
- `Endurance Lab Dashboard.cmd` is placed in the current user's Startup folder because this PC does not allow creation of an `ONLOGON` scheduled task.
- The dashboard listens on port 8501 and starts at the next Windows sign-in.
- Private runtime state and logs remain under `training_data/`.

Use these commands to inspect or remove the automation:

```text
python -m endurance_lab automation-status
python -m endurance_lab automation-remove
```

Running `automation-install` again safely updates the schedule:

```text
python -m endurance_lab automation-install --sync-minutes 60
```

## Phone access without administrator rights

Tailscale requires a system installation and is therefore not usable on this PC without an administrator. The supported no-admin alternative is the portable ngrok agent. The agent makes an outbound encrypted connection; no router or Windows firewall change is required.

The portable executable is stored in the Git-ignored `training_data/tools/ngrok/` directory. Before enabling it:

1. Create an ngrok account.
2. Copy the account authtoken from the ngrok dashboard.
3. Run `python -m endurance_lab phone-access-setup`.
4. Paste the token into the hidden prompt.
5. Save the generated Endurance Lab password when it is displayed; it is shown only once.

The setup writes the ngrok token and Basic Authentication policy under `training_data/auth/`, starts the tunnel, and installs a current-user Startup launcher. Requests without the generated username and password are rejected by ngrok before they reach Streamlit.

Check the current HTTPS address with:

```text
python -m endurance_lab phone-access
```

Do not use an anonymous quick tunnel and do not remove the authentication policy. Treat the ngrok URL and password as private credentials.

## Google Calendar

Calendar publishing uses the official Google Calendar API and a dedicated OAuth authorization. Endurance Lab never receives or stores the Google password.

One-time Google Cloud setup:

1. Create or select a Google Cloud project.
2. Enable the Google Calendar API.
3. Configure the OAuth consent screen for personal use.
4. Create an OAuth client with application type **Desktop app**.
5. Download the JSON file to `training_data/auth/google-calendar-client.json`.
6. Run `python -m endurance_lab calendar-login` and approve access in the browser.
7. Run `python -m endurance_lab calendar-setup --actual strava --planned "Endurance Lab - Planned"`.

The setup locates the existing `strava` calendar, creates the separate planned calendar when needed, and only then enables publishing. Every normal sync sends completed activities from the recent 30-day window to `strava`, while the next 14 days of prescriptions go only to `Endurance Lab - Planned`. Completed activities are transparent timed events; prescriptions are private all-day events with a default reminder 12 hours beforehand. Both flows are idempotent.

Useful commands:

```text
python -m endurance_lab calendar-status
python -m endurance_lab calendar-sync --dry-run
python -m endurance_lab calendar-sync
python -m endurance_lab calendar-activities --from 2026-09-01 --to 2026-09-30 --dry-run
python -m endurance_lab calendar-activities --from 2026-09-01 --to 2026-09-30
```

Calendar synchronization stores the Google event ID for each planned session. Unchanged prescriptions are skipped, changed prescriptions update the same event, and removed future prescriptions remove their linked event.

## Logs and recovery

- `training_data/logs/scheduler.log`: scheduled command output
- `training_data/logs/dashboard.log`: Streamlit startup output
- `training_data/logs/tunnel.log`: portable tunnel output
- `training_data/logs/sync.log`: pipeline stages and download progress

Launcher logs rotate at 5 MB. The database contains sync-stage and calendar-link history. The existing pipeline lock prevents scheduled and manual syncs from changing the manifest or database simultaneously.

The scheduled task runs in the interactive user session. It therefore runs while the Windows user is signed in, including when the screen is locked, but not after signing out or shutting down the computer.
