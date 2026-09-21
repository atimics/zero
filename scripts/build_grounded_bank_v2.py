#!/usr/bin/env python3
"""Build the question-variant dialogue bank for the grounding follow-up.

The first grounded-dialogue bank gave every meaning exactly one question and
one answer. The model learned a usual answer per event and ignored the
question, so eight held-out questions that asked for a different fact of the
same event failed (see experiments/grounded-dialogue/RESULT.md).

This builder keeps the original exchange and adds a second exchange for each
meaning that asks about a different copyable field. The question wording is
deliberately different from the held-out question-contrasts.json probes, so the
follow-up measures question routing rather than memorised wording.

    python3 scripts/build_grounded_bank_v2.py --output \
        experiments/grounded-dialogue/input/dialogue-bank-v2.json
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / 'experiments/grounded-dialogue/input/dialogue-bank.json'
RULES = ROOT / 'experiments/full-event-rehearsal/input/rules.json'

# A second question for meanings whose account holds a second copyable field.
# Answers bind the field number used by rules.json. The reaction is reused from
# the original exchange so the only changed variable is the question's target.
ALTERNATES = {
    'notice_posted_0': ['Where can the notice be found?', 'In {1}.'],
    'character_died_0': ['Who died?', '{0}.'],
    'character_born_0': ['Who was born?', '{0}.'],
    'bakery_production_0': ['Where was the baking done?', 'In {0}.'],
    'paper_milled_0': ['Where is the mill?', 'In {0}.'],
    'woodlot_harvest_0': ['Where is the woodlot?', 'In {0}.'],
    'woodlot_harvest_1': ['Where did they cut?', 'In {0}.'],
    'quarry_output_0': ['Where is the quarry?', 'In {0}.'],
    'masonry_repair_0': ['Where is the work?', 'In {0}.'],
    'masonry_repair_1': ['Where is the work?', 'In {0}.'],
    'masonry_repair_2': ['Where is the work?', 'In {0}.'],
    'sheep_sheared_0': ['Where were the sheep sheared?', 'In {0}.'],
    'cow_slaughtered_0': ['Where were the cattle kept?', 'In {0}.'],
    'cow_calving_0': ['Where did the calving happen?', 'In {0}.'],
    'king_anointed_0': ['Who received the blessing?', '{1}.'],
    'pretender_crisis_0': ['Which order was involved?', '{0}.'],
    'monastic_succession_1': ['Who is the abbot now?', '{0}.'],
    'royal_succession_0': ['Who sought the blessing?', '{0}.'],
    'royal_succession_1': ['Who is the ruler now?', '{0}.'],
    'dragon_patron_named_0': ['Who provided the support?', '{0}.'],
    'dragon_territory_lost_0': ['Whose land was lost?', '{0}.'],
    'royal_carriage_blocked_0': ['Who is travelling?', '{0}.'],
    'royal_carriage_rerouted_1': ['Where is it going?', 'To {0}.'],
    'lore_lost_1': ['What was lost?', '{0}.'],
    'road_site_production_0': ['Where is the work?', 'In {0}.'],
    'prophecy_delivered_0': ['What was delivered?', '{0}.'],
    'horse_bred_0': ['Where were the horses bred?', 'In {0}.'],
    'foal_born_0': ['What was born?', '{3}.'],
    'treasure_crafted_0': ['Where was it crafted?', 'In {0}.'],
    'bandit_pressure_0': ['Who received the help?', '{1}.'],
    'dragon_brood_0': ['Who sealed the hoard?', '{0}.'],
    'bandit_pressure_1': ['Which place had the food?', 'In {3}.'],
    'kingdom_action_0': ['Who acted?', '{0}.'],
    'goblin_cult_rallied_0': ['Who rallied?', '{0}.'],
    'horse_bred_1': ['Where was that?', 'In {0}.'],
    'sheep_bred_0': ['Where was the flock?', 'In {0}.'],
    'sheep_slaughtered_0': ['Where was the slaughter?', 'In {0}.'],
    'harvest_failed_0': ['Which town had the poor harvest?', '{0}.'],
    'route_closed_0': ['Where is the closure?', 'At {0}.'],
    'shortage_1': ['Where was the shortage?', 'In {0}.'],
    'goblin_raided_0': ['Who carried out the raid?', '{0}.'],
    'goblin_raided_1': ['Who carried out the raid?', '{0}.'],
    'settlement_raided_0': ['Where was the settlement?', 'At {1}.'],
    'settlement_raided_1': ['Where was the settlement?', 'At {1}.'],
    'dragon_omen_0': ['Where was the omen seen?', 'In {0}.'],
    'dragon_retaliation_0': ['Who retaliated?', '{0}.'],
    'goblin_cult_rallied_1': ['Who joined the court?', '{0}.'],
    'goblin_cult_rallied_2': ['Who gathered?', '{0}.'],
    'dragon_slain_0': ['Who was named with the dragon?', '{1}.'],
}


def build(original, rules, alternates=ALTERNATES):
    exchanges = {}
    for rule, base in original.items():
        if len(base) != 3:
            raise ValueError(f'{rule} needs question, answer, and reaction')
        variants = [list(base)]
        if rule in alternates:
            question, answer = alternates[rule]
            variants.append([question, answer, base[2]])
        exchanges[rule] = variants
    return {
        'scope': 'Original exchange plus one alternate-fact question per meaning. '
                 'Question wording differs from the held-out question-contrasts probes.',
        'alternates': sorted(alternates),
        'exchanges': exchanges,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    original = json.loads(BANK.read_text())['exchanges']
    rules = {r['id']: r for r in json.loads(RULES.read_text())['rules']}
    if set(original) != set(rules):
        raise ValueError('bank and rules must cover the same meanings')
    missing = sorted(set(original) - set(ALTERNATES))
    a.output.write_text(json.dumps(build(original, rules), indent=2, ensure_ascii=False) + '\n')
    print(f'wrote {a.output} ({len(original)} meanings, '
          f'{len(ALTERNATES)} with alternates; {len(missing)} unchanged)')


if __name__ == '__main__':
    main()