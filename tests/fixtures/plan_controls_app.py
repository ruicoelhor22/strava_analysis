from endurance_lab.db import connect
from dashboard.plan_controls import session_controls

with connect() as connection:
    session_id=connection.execute('SELECT id FROM planned_sessions ORDER BY id LIMIT 1').fetchone()[0]
session_controls(session_id,key_prefix='test-controls')
