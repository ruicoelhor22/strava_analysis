from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.parametrize('action',['SKIP','DRAFT','INCOMPLETE','CANCELLED','NO_PLAN'])
def test_empty_prescription_after_valid_gym_card_does_not_crash_or_publish_targets(action):
    gym={'sport':'strength','title':'Gym A deload','duration_seconds':2400,'description':'Keep gym controlled'}
    original={'sport':'cycling','title':'Threshold touch','duration_seconds':4500,
              'description':'Original threshold prescription','targets':{'power':'280-300 W'}}
    common={'confidence':'high','rules_triggered':[]}
    app=AppTest.from_file(str(Path(__file__).parent/'fixtures'/'coach_cards_app.py'))
    app.session_state['prescriptions']=[
        {**common,'original':gym,'prescribed':gym,'action':'USER_CHOICE'},
        {**common,'original':None if action=='NO_PLAN' else original,'prescribed':None,
         'action':action,'reasons':['This entry has no active workout.']},
    ]
    app.run()
    assert not app.exception
    text=' '.join(item.value for item in app.markdown)
    assert 'Gym A deload' in text and 'No workout prescribed' in text
    assert '280-300 W' not in text and 'Original threshold prescription' not in text
    assert any('no' in item.value.lower() or 'not a confirmed' in item.value.lower() for item in app.caption)
