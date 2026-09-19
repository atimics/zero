"""Build characterful micro-story examples from held accounts, mind state,
occupation voices, and memories.

Each example is a standard micro context: a character's goal, stress, courage,
occupation, memories, and thoughts, their held account with certainty cues,
and recent spoken events. The model continues one line. Instead of abstract
epistemology, the lines are witty, funny, and emotional, grounded in the
character's trade and their memories of the simulation's events.
"""
import copy
import random
from score_crownless_v2 import accepted_forms

QUESTIONS = ['What happened?', 'What have you heard?', 'What is the news?',
             'Tell me what you know.', 'Any word from the roads?', 'What news do you carry?']

# Characterful spoken lines keyed by voice and event family. {actor} and
# {place} are filled from the held account; a memory line may be appended.
VOICE_LINES = {
    'baker': {
        'danger': [
            'First the wheat goes, then the bread follows. The raiders are the only ones eating well.',
            'They took the grain. I suppose that makes them my best customers and my worst.',
            'I bake for a living. The {actor} apparently bake for a living too, only with my ovens.',
        ],
        'scarcity': [
            'No flour, no bread, and no patience left.',
            'You cannot knead dough that does not exist.',
            'I have made bread from worse, but not much worse.',
        ],
        'good': [
            'Finally, the ovens will have something to do.',
            'Good harvests make good bread, and good bread makes good neighbours.',
        ],
        'neutral': ['Bread is the least of it, and the most of it, all at once.'],
    },
    'scribe': {
        'danger': [
            'I will record the raid in the ledger. Under "expenses".',
            'The {actor} have excellent timing. It is always right before I finish the tax rolls.',
            'I shall note the date. If they raid on schedule, we can plan around it.',
        ],
        'scarcity': [
            'The shortage is documented. The food, regrettably, is not.',
            'I have written "famine" so many times it looks like a word again.',
        ],
        'good': [
            'Finally, something worth writing in the good column.',
            'I would engrave this in gold, but the town only has copper.',
        ],
        'neutral': ['It will all be in the records. Whether anyone reads them is another matter.'],
    },
    'farmer': {
        'danger': [
            'The {actor} took the harvest. I grew it, they ate it, the crows get the credit.',
            'Raiders are just a faster kind of weather.',
            'I plant seeds and pray. The {actor} harvest and do not.',
        ],
        'scarcity': [
            'The drought took the harvest, and now the tax collector wants his share of nothing.',
            'My fields gave me nothing this year but a good view of the sky.',
            'You can eat a view of the sky. Once.',
        ],
        'good': [
            'A good year. I had forgotten what the fields look like when they smile.',
            'This year the land keeps its promises.',
        ],
        'neutral': ['Farming is just gambling with better odds and worse weather.'],
    },
    'smith': {
        'danger': [
            'They took the tools. I will make more, but the {actor} will have to bring their own next time.',
            'A raid means new work for me. I would rather be unemployed.',
            'I forge the blades and the {actor} sharpen them on my customers.',
        ],
        'scarcity': [
            'No ore, no iron, no nails. The town will fall apart one board at a time.',
            'I can fix a plow, but I cannot fix an empty forge.',
        ],
        'good': [
            'Good news. Now I can sell the good iron instead of hoarding it.',
            'When the town prospers, the anvil sings.',
        ],
        'neutral': ['Every hammer blow is a prayer that the work holds.'],
    },
    'innkeeper': {
        'danger': [
            'The {actor} took the grain. The guests took the ale. I am left with the stories.',
            'Business is booming. Everyone wants a drink after the raid.',
            'I keep the door open and the good wine hidden. The {actor} get the cheap stuff.',
        ],
        'scarcity': [
            'No food, no guests, no coin. The inn is just a large, cold house.',
            'I can water the ale, but I cannot water the bread.',
        ],
        'good': [
            'Good news fills the common room faster than bad ale empties it.',
            'When the town smiles, the innkeeper eats.',
        ],
        'neutral': ['Every traveller has a story, and every story wants a drink.'],
    },
    'miller': {
        'danger': [
            'They took the grain before it reached my stones. The flour is safe, for now.',
            'The {actor} raid the fields, not the mill. They have no taste for dust.',
        ],
        'scarcity': [
            'No grain, no flour, no bread. The mill grinds air.',
            'I can mill anything with a kernel. I cannot mill nothing.',
        ],
        'good': [
            'A full cart of grain is a beautiful thing.',
            'The millstone turns, and the town eats. Today it turns well.',
        ],
        'neutral': ["The millstone hears all the town's secrets and grinds them to flour."],
    },
    'shepherd': {
        'danger': [
            'The wolves take one sheep, the {actor} take the flock. I am left with the dog.',
            'My flock is smaller, but the dog and I are still counting.',
        ],
        'scarcity': [
            'Thin sheep, thin wool, thin winter.',
            'The flock is lean, and the pasture is leaner.',
        ],
        'good': [
            'A fat lamb is the best news a shepherd can hear.',
            'The flock grows, and so does my heart.',
        ],
        'neutral': ['Sheep are simple. People are the complicated ones.'],
    },
    'woodcutter': {
        'danger': [
            'They took the woodpile. I will cut more, but the {actor} can carry it next time.',
            'An axe is a tool. In a raid, it is a tool and a friend.',
        ],
        'scarcity': [
            'No wood, no fires, no warm winter.',
            'The forest gives, but it will not give twice.',
        ],
        'good': [
            'Good timber and good news. The forest smiles.',
            'When the town builds, I eat.',
        ],
        'neutral': ['Every tree falls, and every fall feeds someone.'],
    },
    'quarryman': {
        'danger': [
            'They took the stone. Good luck carrying a quarry to a raid.',
            'The {actor} want stone now? They will build a wall and thank us later.',
        ],
        'scarcity': [
            'No stone, no repairs, no walls. The town leans.',
            'I can cut stone, but I cannot cut food from it.',
        ],
        'good': [
            'Good news means new walls, and new walls mean work.',
            'When the town builds, the quarry feeds its people.',
        ],
        'neutral': ['Stone does not lie, and it does not hurry.'],
    },
    'cartwright': {
        'danger': [
            'They took the wheels. I will make more, but the {actor} can walk next time.',
            'A cart without wheels is just a heavy box. The {actor} made many heavy boxes.',
        ],
        'scarcity': [
            'No wood, no wheels, no trade. The roads are just paths again.',
            'I can build a cart, but I cannot build the food to fill it.',
        ],
        'good': [
            'Good roads and good carts move the whole valley.',
            'When trade returns, my wheels turn again.',
        ],
        'neutral': ['Every cart carries a story, and every story wears out a wheel.'],
    },
    'official': {
        'danger': [
            'The {actor} raid, the town burns, and I file the report.',
            'Order is a thin line between the {actor} and the mob. I stand on it.',
            'I keep the peace. The {actor} keep me busy.',
        ],
        'scarcity': [
            'The granary is empty and the paperwork is full.',
            'I can write a ration decree, but I cannot ration nothing.',
        ],
        'good': [
            'Good news makes governing almost pleasant.',
            'When the town prospers, the office runs itself.',
        ],
        'neutral': ['Paperwork is the true tax. Everything else is negotiable.'],
    },
    'courier': {
        'danger': [
            'I carry the news and the {actor} try to stop it. The news is faster.',
            'The road is dangerous, but the message is more dangerous if it does not arrive.',
        ],
        'scarcity': [
            'I carry word of the shortage, not the food itself. Both would be better.',
            'My saddlebags hold letters, not bread. The letters are colder.',
        ],
        'good': [
            'Good news is light to carry.',
            'I have carried worse. This one almost runs by itself.',
        ],
        'neutral': ['The road knows my name by now.'],
    },
    'refugee': {
        'danger': [
            'I left everything to the {actor}. I kept my feet, and that is all.',
            'Home is where the {actor} have not been yet.',
        ],
        'scarcity': [
            'I have eaten less than I should and walked more than I can.',
            'Hunger is a companion I did not choose.',
        ],
        'good': [
            'A kind word and a warm place. That is more than I had.',
            'Maybe this town will hold.',
        ],
        'neutral': ['I am a guest here, and I try to be a light one.'],
    },
    'scout': {
        'danger': [
            'I saw the {actor} coming and told the town. They thanked me by asking what took so long.',
            'Scouting is easy. Being believed is the hard part.',
        ],
        'scarcity': [
            'I have walked the empty fields. The roads are the only full thing.',
            'The land is thin, and I have seen it all.',
        ],
        'good': [
            'I saw the good news coming from a day away.',
            'A quiet road is a good road.',
        ],
        'neutral': ['I walk ahead so the rest can walk safely.'],
    },
    'traveller': {
        'danger': [
            'I have crossed worse roads. The {actor} made this one memorable.',
            'Every inn has a story about the {actor}. I am collecting them.',
        ],
        'scarcity': [
            'I have seen empty granaries in three towns. This is the fourth.',
            'The road is long and the meals are short.',
        ],
        'good': [
            'This town is a good place to rest my feet.',
            'I will tell the next town about this good news.',
        ],
        'neutral': ['The road is my home, and every town is a room in it.'],
    },
    'laborer': {
        'danger': [
            'I work for my bread, and the {actor} take it before I earn it.',
            'Hard work feeds me. The {actor} feed on hard work.',
        ],
        'scarcity': [
            'No work, no bread, no rest.',
            'I have worked all day for a half-day\u2019s food.',
        ],
        'good': [
            'Work is good. Work means food.',
            'When the town builds, I eat well.',
        ],
        'neutral': ['My hands are my fortune, and they are worn thin.'],
    },
    'bandit': {
        'danger': [
            'Hungry men make the best recruits. They do not ask questions, they just march.',
            'We are not thieves. We are a redistribution service.',
            'The {actor} raid with swords. We raid with patience. Patience pays better.',
        ],
        'scarcity': [
            'A lean year for the towns is a lean year for us. We all eat from the same pot.',
            'We take what we need. In a bad year, we need a lot.',
        ],
        'good': [
            'A fat town is a fat target. Good news for us.',
            'When the towns prosper, our work gets easier.',
        ],
        'neutral': ['The road takes its toll, and so do we.'],
    },
    'goblin': {
        'danger': [
            'We take the wheat, the town takes the blame. Everyone is happy.',
            'The dragon gets the tribute, we get the leftovers, and the town gets the tax.',
            'Raiding is just trade with better timing.',
        ],
        'scarcity': [
            'A hungry town is a quiet town. A hungry goblin is a loud one.',
            'We raid for bread. The dragon raids for gold. We are the honest ones.',
        ],
        'good': [
            'A fat town is a fat tribute. The dragon will be pleased.',
            'Good harvests mean good tribute, and good tribute means the dragon sleeps.',
        ],
        'neutral': ['We keep the accounts in our heads. The scribes keep them in books, and lose them.'],
    },
    'resident': {
        'danger': [
            'The {actor} came, took what they wanted, and left. I am still counting what remains.',
            'We heard the {actor} were coming. We hoped they were not.',
        ],
        'scarcity': [
            'The stores are low and the winter is long.',
            'We make do with less, and the less keeps shrinking.',
        ],
        'good': [
            'Good news. We needed some.',
            'It is good to hear something that is not bad.',
        ],
        'neutral': ['We keep our heads down and our doors shut.'],
    },
}

# Lines that connect the current event to a remembered one. {memory} is filled
# with the character's held memory line.
MEMORY_LINES = {
    'danger': [
        'Last time the {actor} came, we lost more than this. I remember it well.',
        'I remember when this was a quiet road. Now every raid is a memory.',
        'The last raid taught us to hide the good things first. This time we were ready.',
        'I still remember the old raid. This one has the same smell.',
    ],
    'scarcity': [
        'The last shortage taught us to count every ear. This one is worse.',
        'I remember the lean years. This feels like one of them.',
        'We survived the last famine. I hope we remember how.',
    ],
    'good': [
        'The last good year was too long ago. I had forgotten how it felt.',
        'I remember the old days, before the troubles. This feels like one of them.',
        'Good news is rare enough that I remember the last one exactly.',
    ],
    'neutral': [
        'I have seen this before, and it did not end the way we hoped.',
        'We have been through this. We will be through it again.',
    ],
}

# Characterful internal thoughts keyed by goal and event family.
THOUGHT = {
    'secure_livelihood': {
        'danger': [
            'If the {actor} come again, my family will starve.',
            'I should hide what little we have before they return.',
            'Everything I have saved could be gone in a day.',
        ],
        'scarcity': [
            'I need to find food before the stores run dry.',
            'There is not enough bread to last the winter.',
            'Every loaf counts while food is short.',
        ],
        'good': [
            'At last, something to be thankful for.',
            'This takes a weight off my shoulders.',
            'Good news means a safer season ahead.',
        ],
        'neutral': ['I should keep an eye on how this turns out.'],
    },
    'survive_crisis': {
        'danger': [
            'I should find shelter before nightfall.',
            'The {actor} could be here any day. I have to be ready.',
            'I have to keep my family out of the way of this.',
        ],
        'scarcity': [
            'I have to ration what we have.',
            'Every scrap counts now.',
            'We cannot afford to waste anything this season.',
        ],
        'good': [
            'Maybe things are turning around.',
            'A little good news goes a long way.',
            'This gives us a chance to recover.',
        ],
        'neutral': ['We will get through this somehow.'],
    },
    'carry_news': {
        'danger': [
            'This news must reach the next town before the {actor} do.',
            'Someone has to warn the other settlements.',
            'People need to know what is coming.',
        ],
        'scarcity': [
            'People need to know how bad the harvest was.',
            'The next town must hear about the shortage.',
            'If no one carries the word, towns will starve quietly.',
        ],
        'good': [
            'This is the kind of news people want to hear.',
            'Good news travels fast; I should spread it.',
            'The whole valley should know about this.',
        ],
        'neutral': ['I will carry this word and see where it leads.'],
    },
    'keep_order': {
        'danger': [
            'The town needs to post guards before the next raid.',
            'We must keep order or the panic will do more harm than the {actor}.',
            'Someone has to keep the town calm through this.',
        ],
        'scarcity': [
            'The stores must be guarded or people will riot.',
            'We need to share the food fairly or there will be trouble.',
            'Rationing has to be seen as fair or the town will split.',
        ],
        'good': [
            'Good news will settle the town down.',
            'This will lift everyone\u2019s spirits.',
            'A steady hand now keeps the peace later.',
        ],
        'neutral': ['A steady hand keeps the peace.'],
    },
}

STRESS_PREFIX = {
    'high': ['My hands are shaking. ', 'I can barely think. ', 'My heart is pounding. '],
    'medium': ['I need to stay calm. ', 'I should think this through. ', ''],
    'low': ['Perhaps it will pass. ', 'I will keep calm. ', ''],
}

DANGER = {'GOBLIN_RAIDED', 'SETTLEMENT_RAIDED', 'BANDIT_PRESSURE',
          'BANDIT_RAID_DEPARTED', 'BANDIT_RAID_RETURNED', 'DRAGON_BROOD',
          'DRAGON_OMEN', 'DRAGON_RETALIATION', 'DRAGON_TERRITORY_LOST',
          'DRAGON_SLAIN', 'DRAGON_BATTLE', 'DRAGON_MUSTERED', 'COURIER_LOST',
          'WAR_DECLARED', 'GOBLIN_RAID_DEPARTED', 'DRAGON_HOARD_STOLEN',
          'DRAGON_TREASURE_RETURNED', 'DRAGON_HOARD_DEFENDED'}
SCARCITY = {'SHORTAGE', 'HARVEST_FAILED', 'SHEEP_SLAUGHTERED', 'COW_SLAUGHTERED'}
GOOD = {'NOTICE_POSTED', 'PEACE_DECLARED', 'ALLIANCE_DECLARED', 'TREASURE_CRAFTED',
        'SHEEP_BRED', 'COW_CALVING', 'BAKERY_PRODUCTION', 'QUARRY_OUTPUT',
        'WOODLOT_HARVEST', 'SHEEP_SHEARED', 'PAPER_MILLED', 'HORSE_BRED',
        'FOAL_BORN', 'GOBLIN_TRADE', 'COURIER_ARRIVED', 'COURIER_DEPARTED',
        'CHARACTER_BORN', 'KING_ANOINTED'}

CERTAINTY = ['How sure are you?', 'Are you sure?', 'Can we trust that account?']
CHECK = 'We should ask someone who was there.'
ASK = "Let's ask around before we pass it on."
CLOSE = "Agreed. Let's leave it there for now."
# Nine conversation acts and four mind acts. The mind work replaced the first
# family rather than adding to the second, which is how the model lost the
# ability to hold an exchange. A corpus that serves the runtime needs both:
# CcCoreModelBegin asks for a conversation turn, CcCoreModelBeginMind for a
# mind turn, and one model has to answer either.
CONVERSATION_ACTS = ['start', 'question', 'agree', 'disagree', 'certainty', 'source',
                     'check', 'close', 'end']
MIND_ACTS = ['say', 'memory', 'thought', 'react']
ACTS = CONVERSATION_ACTS + MIND_ACTS


def family(kind):
    if kind in DANGER: return 'danger'
    if kind in SCARCITY: return 'scarcity'
    if kind in GOOD: return 'good'
    return 'neutral'


def mind_row(row):
    return row.get('mind', {'goal': 'secure_livelihood', 'stress': 'medium',
                            'courage': 'medium', 'memories': [], 'thoughts': []})


def fill(text, row):
    values = {}
    for field in row['fields']:
        if field['role'] in (1, 3): values.setdefault(field['role'], field['text'])
    return text.format(actor=values.get(1, 'raiders'), place=values.get(3, 'the towns'))


def authored_thought(row, rng):
    mind = mind_row(row)
    pool = THOUGHT.get(mind['goal'], {}).get(family(row.get('kind')),
                    ['I should keep an eye on how this turns out.'])
    return fill(rng.choice(pool), row)


def response(base, rule, act, rng, replacement=None, paraphrase=False):
    row = copy.deepcopy(base)
    own = row['output']
    heard = rng.choice(sorted(accepted_forms(row, rule)))
    mind = mind_row(row)
    if act in MIND_ACTS:
        row['mind'] = mind
    goal = mind['goal']
    fam = family(row.get('kind'))
    voice = row.get('voice', 'resident')
    if voice not in VOICE_LINES: voice = 'resident'
    history = []
    prefix, suffix = '', ''
    use_claim = False
    changed = None
    if act in CONVERSATION_ACTS:
        # A conversation turn is answered from the account alone, in the shape
        # CcCoreModelBegin emits: no mind lines, no control cue.
        row.pop('mind', None); row.pop('voice', None)
        candidates = [f for f in row['fields'] if f.get('spoken') and f['knowledge'] != 3 and
                      f['text'] in heard and f['role'] not in (0, 8)]
        if act == 'disagree' and candidates:
            field = rng.choice(candidates)
            alternative = replacement(field) if replacement else 'Farhaven'
            if alternative == field['text']: alternative = 'Another place'
            heard = heard.replace(field['text'], alternative)
            changed = {'field': field['field'], 'own': field['text'], 'heard': alternative}
        elif act == 'disagree': act = 'agree'
    if act == 'start': target, use_claim = own, True
    elif act == 'question':
        history = [{'speaker': 'other', 'text': 'Tell me the news.' if paraphrase else rng.choice(QUESTIONS)}]
        target, use_claim = own, True
    elif act == 'agree':
        history = [{'speaker': 'other', 'text': heard}]
        target = "That's what I heard too. How sure are you?"
    elif act == 'disagree':
        history = [{'speaker': 'other', 'text': heard}]
        prefix, suffix = 'I heard a different account. ', ' How sure are you?'
        target, use_claim = prefix + own + suffix, True
    elif act == 'certainty':
        history = [{'speaker': 'self', 'text': own}, {'speaker': 'other', 'text':
            'How certain is that?' if paraphrase else rng.choice(CERTAINTY)}]
        if not paraphrase and rng.randrange(2):
            history[-1]['text'] = 'I heard a different account. ' + heard + ' ' + history[-1]['text']
        target = ('I am unsure. ' if row['confidence'] < 40 else
                  'The account has passed through several people. ' if row['retold'] else
                  'That is the account I hold. ') + CHECK
    elif act == 'source':
        history = [{'speaker': 'self', 'text': own}, {'speaker': 'other', 'text':
                   'Where did that story come from?' if paraphrase else 'Who told you?'}]
        target = 'That is the account I hold. ' + CHECK
    elif act == 'check':
        history = [{'speaker': 'self', 'text': rng.choice(CERTAINTY)},
                   {'speaker': 'other', 'text': rng.choice(['I am unsure. ', 'That is the account I hold. ',
                       'The account has passed through several people. ']) + CHECK}]
        target = ASK
    elif act == 'close':
        history = [{'speaker': 'self', 'text': CHECK}, {'speaker': 'other', 'text': ASK}]
        target = CLOSE
    elif act == 'end':
        history = [{'speaker': 'self', 'text': ASK}, {'speaker': 'other', 'text': CLOSE}]
        target = 'Very well.'
    elif act == 'say':
        history = [{'speaker': 'other', 'text': heard}]
        pool = VOICE_LINES[voice].get(fam, VOICE_LINES[voice]['neutral'])
        target = fill(rng.choice(pool), row)
    elif act == 'memory':
        history = [{'speaker': 'other', 'text': heard}]
        pool = MEMORY_LINES.get(fam, MEMORY_LINES['neutral'])
        line = fill(rng.choice(pool), row)
        if mind.get('memories'):
            memory = rng.choice(mind['memories'])
            if len(memory) > 80: memory = memory[:80].rsplit(' ', 1)[0] + '...'
            connector = rng.choice([' It reminds me of when ', ' It puts me in mind of ',
                                    ' Like when ', ' It brings back '])
            target = line + connector + memory
        else:
            target = line
    elif act == 'thought':
        history = []
        thought = authored_thought(row, rng)
        if mind.get('stress') == 'high' and rng.randrange(2):
            thought = rng.choice(['My hands are shaking. ', 'I can barely think. ']) + thought
        # The control cue already names the act; repeating the marker in the
        # target teaches the model to speak its own scaffolding.
        target = thought
    elif act == 'react':
        history = [{'speaker': 'other', 'text': heard}]
        row['mind']['stress'] = 'high'
        pool = VOICE_LINES[voice].get(fam, VOICE_LINES[voice]['neutral'])
        target = fill(rng.choice(pool), row)
    else: raise ValueError(act)
    # Conversations reach the model several turns deep and the encoder keeps
    # the last four. Stacking exchanges here is what teaches the model to hold
    # together at that depth instead of only answering an opening question.
    for _ in range(rng.choice((0, 1, 1, 2)) if history else 0):
        first = history[0]['speaker']
        second = 'self' if first == 'other' else 'other'
        history = [{'speaker': first, 'text': rng.choice(QUESTIONS)},
                   {'speaker': second, 'text': own if second == 'self' else heard}] + history
    row.update(output=target, history=history, act=act, changed=changed,
               control={'thought': 'think', 'memory': 'remember'}.get(act, 'say'))
    if use_claim:
        offset = len(prefix.encode())
        for span in row['copies']:
            span['start'] += offset
            span['end'] += offset
    else: row['copies'] = []
    return row


def build_rows(bases, rules, seed, repeats=1, paraphrase=False, acts=None):
    rng = random.Random(seed)
    pools = {}
    for row in bases:
        for f in row['fields']:
            if f.get('spoken') and f['knowledge'] != 3: pools.setdefault(f['role'], set()).add(f['text'])
    pools = {role: sorted(values) for role, values in pools.items()}
    def replacement(field):
        options = [x for x in pools[field['role']] if x != field['text']]
        return rng.choice(options) if options else 'A different account'
    result = []
    # Mind acts need a character's goal and stress; conversation acts answer
    # from the account alone. Callers pick the family that suits their rows.
    chosen = acts or ACTS
    for repetition in range(repeats):
        for i, base in enumerate(bases):
            # Adjacent source-value pairs receive the same response task.
            act = chosen[(i // 2 + repetition * 4) % len(chosen)]
            row = response(base, rules[base['rule']], act, rng, replacement, paraphrase)
            row['id'] += f':conversation:{repetition}'
            result.append(row)
    return result


def approved_response(row, rule):
    if row['act'] in ('start', 'question'):
        return accepted_forms(row, rule)
    if row['act'] == 'disagree':
        return {'I heard a different account. ' + form + ' How sure are you?'
                for form in accepted_forms(row, rule)}
    return {row['output']}
