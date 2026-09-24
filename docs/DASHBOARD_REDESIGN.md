# Responsive performance dashboard

## Audit and information architecture

The previous interface placed twelve destinations in the sidebar. The Home equivalent started with six equally weighted KPIs; the activity library and current week opened as wide tables. Coach decisions, training cost, and the next key workout required scrolling past secondary metrics. Sport and load pages retained useful charts but presented many without a clear question.

| Decision | Area | Result |
| --- | --- | --- |
| Keep | Dark performance palette, plots, route, zones, load model, comparison data | Analytical depth remains available |
| Improve | Today's session, week, activity summary, Coach reasoning | Primary content appears before raw metrics |
| Move | System status and configuration | Secondary sidebar destination |
| Collapse | Filters, full tables, zones, halves, comparison data, detailed cost | Details remain accessible without dominating phones |
| Remove from default | Repeated full tables, six equal Home KPIs, twelve primary destinations | Faster scanning and less vertical clutter |

Primary navigation is Home, Plan, Activities, Performance, Coach. System & data is secondary. Performance includes Overview, Load, Cycling, Running, Swimming, Strength, and Advanced. The original detailed overview and coaching evidence live in Advanced.

## Components and tokens

`dashboard/_shared.py` owns reusable page headings, semantic badges, metric tiles, sport markers, activity cards, cost dimensions, formatting, and chart styling. `dashboard/styles.css` holds spacing, backgrounds, type scale, borders, semantic status colors, sport colors, and breakpoints at 900 and 600 px. The layout uses flat, bordered instrument panels with tabular numbers.

Home reads derived summaries, prescriptions, plan matches and a small number of persisted cost records. It does not load activity streams. Detailed streams load only when a session is opened. The activity feed shows 18 records per page; its full table is expandable. The Plan week uses seven compact desktop columns, two tablet columns, and vertical day rows on a phone.

## Validation and limitations

All existing Python tests and compilation checks passed. Streamlit's test runner rendered Home, Plan, Activities, Performance, Coach, the sport/deep-analysis choices, and an actual activity detail without application exceptions. The browser-control connection was unavailable in this session, so 390, 768 and 1440 px visual inspection, screenshot comparison, and browser-console checks still require a connected browser. CSS is designed to contain tables and charts and stack column blocks on narrow screens, but those visual criteria are not claimed as verified.

The redesign does not change ingestion, analytics, coaching decisions, plan matching, or the database schema.
