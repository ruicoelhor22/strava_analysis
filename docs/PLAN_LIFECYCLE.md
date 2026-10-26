# Plan lifecycle and athlete control

The coach workbook is read-only. Endurance Lab keeps four distinguishable layers: what the coach wrote, what the app recommended, what the athlete chose, and what was completed. A saved athlete choice never replaces the independent app prescription.

## Revisions

Normal sync checks the configured/imported workbook path using SHA-256. An unchanged file skips parsing and plan writes after the one-time legacy baseline. A changed file is read as one immutable byte snapshot, parsed and validated before a transactional import. Failed parsing leaves the last valid plan active, records `PLAN_IMPORT_FAILED`, and allows activity processing to continue.

`plan_revisions` holds hashes, import times, ranges, review state and compact changes. `plan_session_versions` preserves session instructions; the existing `training_plan_sources` remains the workbook provenance archive. No second workbook-blob store is added. The additive lifecycle tables work with the existing schema and preserve activities, sources, matches and prescription history.

Identity matching first uses source keys, then defensible unique workout codes, same-date/sport edits and near-identical one-day moves. Ambiguous edits deliberately remain additions/removals; the app does not guess a correspondence. Supported differences include additions, removals, dates, sports, durations, intensity/targets, notes, priorities, phases, week types and source status. Session-level phase labels take precedence over a week's majority label at phase boundaries.

Activity matching also recognizes same-week long aerobic ride intent when measured power-zone evidence and duration agree. A several-days-early ride is surfaced as a possible reordered endurance session, not silently credited against a future key ride: extra training and replacement training have different meanings. Threshold work cannot be satisfied by duration alone. The Plan page separately shows actual training time, linked completion and extra/reordered training. Explicit moved gym choices retain session identity and are linked once the activity is actually recorded.

Actual activities appear directly on their performed-day cards even when no coach workout was scheduled. An empty coach slot does not mean no training occurred. Athlete-confirmed extra training is persistently classified in the audit log and excluded from automatic plan matching; it still counts toward volume and coaching cost. Explicitly accepted alternative sports can complete the chosen session without pretending the original sport was performed. Weekly summaries distinguish plan completion from the number of actual recorded activities.

Matched historical sessions and workbook-recorded completions retain their original records. Revised coach instructions remain visible in revision history. Future records can change while preserving identity. Material edits retire stale app recommendations and flag saved athlete decisions for review. Small title typos and athlete-note edits do not invalidate a decision; workout structure, target, duration, date and sport changes do.

Use `plan-import --dry-run --show-diff` to preview without activating a revision, `--force` to explicitly revalidate/reimport an unchanged file, and `plan-status` for revision, lifecycle and conflict counts. Keep the workbook at its existing path. Replacing or renaming the source file is not automatically treated as the same plan.

## Lifecycle and precedence

- **Confirmed:** explicitly scheduled/approved and sufficiently specified.
- **Draft:** usable content without explicit confirmation. The athlete can confirm it in the app.
- **Incomplete:** missing workout/sport/duration or placeholder instructions. Modify explicitly or wait for coach revision; the app does not invent missing content.
- **Completed:** recorded completion, with measured evidence distinguished from workbook-only metadata.
- **Cancelled:** explicitly skipped/cancelled or removed by the coach.

The existing workbook's `Planned` status counts as explicit scheduling only when usable details exist. A phase outline on otherwise blank dated rows is not a workout. Blank calendar days are reported separately, not inflated into hundreds of fictitious planned sessions. Rest choices, skips, drafts and incomplete entries do not count as missed training debt or firm planned-volume commitments.

Effective workout precedence is athlete decision, independent active app recommendation, confirmed coach instructions, then a non-prescriptive draft/incomplete indication. A withdrawn session with a saved decision remains visible as a conflict. Its choice is retained until the athlete follows the removal or explicitly retains/modifies it.

Today, Plan and Coach share compact controls for Accept prescription, Keep original, Modify, Move, Skip and Clear decision. Modifications cover date, sport, duration, intensity and description, plus an optional note. Moves retain the session ID. Decision metadata and append-only history record the action, source, timestamps, revision basis and previous/resulting effective states. Clearing a choice retains its history. Completing a session locks editing.

When a choice already exists, the default action is **Keep my decision**. This retains its exact saved workout, including Rest or a previously accepted recommendation, without recalculating it. **Accept app recommendation** explicitly shows the independent app workout and requires confirmation before replacing a different saved choice. It accepts the displayed recommendation, not a newly calculated workout. **Use original coach plan** is deliberately labelled as returning to the coach's instructions, not keeping the current athlete choice.

## Phase modifiers

Original coach terminology is retained. Conservative mappings recognize BASE, BUILD, RECOVERY, TAPER, RACE and UNKNOWN. Week type is separate: a BUILD phase may contain a recovery/deload week.

- Base protects consistency and can ease optional quality after repeated high-cost work.
- Build protects quality anchors and can omit interfering optional support rather than create replacement volume.
- Recovery/deload explains intentional reduced volume; no compensatory training is added.
- Taper/race protects freshness and can omit lower/full-body support strength near a protected event.
- Explicit race sessions receive A priority; post-race support is recovery-gated.
- Unknown labels retain the general engine. Ambiguous pre-/post-holiday labels are not forced into a recovery phase.

These are modifiers to the existing cost/interference/priority/runway rules, not separate engines, new metrics or an autonomous plan generator. The phase text, week type and reasons are included in decision traces.

## Calendar and events

Explicitly requested app proposals can fill a dated gap without modifying the coach workbook. `app_planning.save_app_plan` validates one workout/rest entry per date, travel modality restrictions and no overlap with existing active sessions. It saves a separate, labelled app-proposal source, lifecycle snapshots and audit events. Repeating the same proposal is a no-op; edited proposals require explicit revision review rather than silent replacement. Coach-workbook refresh continues to use the coach source, not the supplemental JSON. Navigation keeps the combined active date range. A later coach plan overlapping an app proposal needs review; this is not automatic full-season plan generation.

Zero-duration, explicitly labelled rest entries remain rest, including after key workouts; they are not conditional workouts requiring a readiness gate.

An explicitly confirmed intervening rest day allows moderate, non-high-systemic strength interference to produce a gentle maintenance recommendation instead of automatically cancelling gym. High muscular cost and stronger fatigue gates remain conservative. This does not establish readiness: demanding or sore-muscle work is omitted, and no compensatory sets are added.

App and calendar resolve the same serialized effective prescription. Planned sync reconciles the previous seven days as well as the upcoming horizon, correcting late rest confirmations without touching unrelated calendar events.

The planned calendar resolves the effective workout directly. Draft/incomplete entries are excluded by default; explicit usable athlete choices may be published. Moves update the linked event, including moving an existing event outside the usual future horizon. Skips remove future planned events. Clearing a skip can recreate an event without attempting to update a deleted remote event. Completed-activity calendar logic is unchanged.

Descriptions briefly identify coach instructions, app action and athlete choice. Material conflicts are labelled without publishing an entire trace. Content hashes keep updates idempotent. UI decisions save locally first; failed calendar updates are retried during normal sync.

`plan_events` emits lightweight structured revision, conflict, prescription-change, missing-confirmed-session and sync-failure events. This prepares notifications without adding a notification delivery service or message broker.

## Historical validation and limits

Backtests use the latest revision known before each local decision day, decision history known then, and prior completed activity data. Later check-in edits are excluded rather than guessed. Missing revision history is not reconstructed from today's workbook. A migrated baseline preserves the existing import timestamp; it does not claim a complete historical archive of earlier coach versions.

Phone controls use responsive cards, stacked inputs and full-width save buttons. Automated Streamlit tests validate rendering and decision persistence. A connected browser is required for an actual 390px visual inspection; automated tests alone do not prove visual quality.

Validation does not establish physiological readiness or medical safety. The current real-data historical sample is small, and a high adjustment rate should be reviewed before granting greater autonomy. Always-on hosting and notification delivery remain separate future phases.
