"""Three-axis dialogue for the Crownless core: move, channel, stance.

The flat thirteen-act list answered three unrelated questions at once, and the
collision cost it variety: five acts carried exactly one sentence between 5,500
rows each, so a baker, a smith and a scribe all closed a conversation with
"Very well." Splitting the axes makes the missing cells reachable.

    move     what the turn does          open answer remark affirm dispute
                                         hedge attribute defer settle part
                                         recall muse
    channel  where the line goes         spoken thought remembered
    stance   who is speaking             voice goal stress courage

`react` is gone as an act: it was `remark` at high stress, which the stance
axis now carries on its own.

One rule governs the phrasing tables. Variety is keyed, never free: a pool is
chosen by state the prompt can see, and the free choice inside a cell is two or
three near-synonyms. Wide across cells, narrow within. Random choice across
unrelated wordings leaves the target underdetermined and the model learns their
average, which is how character turns to mush.

`forms()` is the single source of truth. The generator picks from it and the
grader accepts all of it, so wording and scoring cannot drift apart.
"""
import random

# Each move names the channel it is normally produced on. Channel stays a
# separate axis so a move can be re-voiced through another one deliberately.
MOVES = {
    'open': 'spoken', 'answer': 'spoken', 'remark': 'spoken', 'affirm': 'spoken',
    'dispute': 'spoken', 'hedge': 'spoken', 'attribute': 'spoken', 'defer': 'spoken',
    'settle': 'spoken', 'part': 'spoken', 'recall': 'remembered', 'muse': 'thought',
}

# The trailing cue names the MOVE, not the channel.
#
# Cueing the channel instead puts ten of the twelve moves behind one `# say:`
# and leaves the move axis in the data but absent from the input: six prompt
# shapes then map to four or five different targets and the model can only
# guess. Measured, that pinned held-out accuracy at 12/48 through 5,500 steps
# while the loss halved. `think` and `remember` were the only cues that worked
# precisely because those two are move names already.
CUE = {move: move for move in MOVES}
VOICES = ('baker', 'scribe', 'farmer', 'smith', 'innkeeper', 'miller', 'shepherd',
          'woodcutter', 'resident')
STRESS = ('low', 'medium', 'high')

# The nine recovered conversation acts map onto moves one for one, so the
# behaviour the old corpus taught survives as a subset of this one.
FROM_ACT = {'start': 'open', 'question': 'answer', 'agree': 'affirm', 'disagree': 'dispute',
            'certainty': 'hedge', 'source': 'attribute', 'check': 'defer', 'close': 'settle',
            'end': 'part', 'say': 'remark', 'react': 'remark', 'memory': 'recall',
            'thought': 'muse'}

# --- affirm: corroborate what the other speaker just said -------------------
AFFIRM = {
    'baker': {'low': ["That is the word over the counter too.", "Same as I heard at the bakehouse."],
              'medium': ["I have heard the same, and it sits badly.", "That matches what came in with the flour."],
              'high': ["I heard it too, and I have not slept since.", "The same, and I am short of bread already."]},
    'scribe': {'low': ["That agrees with what I have set down.", "My record says the same."],
               'medium': ["It matches my entry, near enough.", "I wrote as much, in those words."],
               'high': ["It matches, and I wish it did not.", "My own hand says the same. That frightens me."]},
    'farmer': {'low': ["That is how it reached my field.", "Same story came down the lane."],
               'medium': ["I heard that, and the soil agrees.", "That is the account that reached us."],
               'high': ["I heard it, and the season will not bear it.", "The same, and we have nothing spare."]},
    'smith': {'low': ["That came through the forge too.", "Same as I had it."],
              'medium': ["I heard that, and I believe it.", "It matches what the carters said."],
              'high': ["I heard it. I have been sharpening since.", "The same. I will bar the door tonight."]},
    'innkeeper': {'low': ["Half the room said as much last night.", "That is the tale at my tables."],
                  'medium': ["I have heard it three times this week.", "Same, and from better men than most."],
                  'high': ["Everyone says it, and they say it frightened.", "I have heard it all night. It does not improve."]},
    'miller': {'low': ["The same came up with the grain.", "That is how it reached the mill."],
               'medium': ["I heard that, and the sacks bear it out.", "Same account, near enough."],
               'high': ["I heard it, and the stones will stand idle.", "The same. There will be nothing to grind."]},
    'shepherd': {'low': ["That is what they said on the hill.", "Same as reached the pasture."],
                 'medium': ["I heard it out with the flock.", "That matches what the drovers told me."],
                 'high': ["I heard it, and I brought them in early.", "The same. I have counted them twice since."]},
    'woodcutter': {'low': ["That reached the wood as well.", "Same as I had it on the road."],
                   'medium': ["I heard as much coming down.", "It matches what the carters carried."],
                   'high': ["I heard it, and I keep the axe near.", "The same. I do not walk out alone now."]},
    'resident': {'low': ["That is what I heard too.", "Same as reached me."],
                 'medium': ["I have heard the same.", "That matches my account."],
                 'high': ["I heard it too, and it worries me.", "The same, and I like it no better."]},
}

# --- defer: propose that someone better placed be asked ---------------------
DEFER = {
    'baker': {'low': ["Ask at the mill. They hear it before I do.", "The carters will know better than me."],
              'medium': ["We should ask someone who was there.", "Better to hear it from someone closer."],
              'high': ["Find someone who saw it, before we spread it further.", "Ask someone who was there. I will not repeat it otherwise."]},
    'scribe': {'low': ["I would want it from a witness before I write it.", "Let us have it from someone who saw."],
               'medium': ["We should ask someone who was there.", "I would not enter it on hearsay."],
               'high': ["I will not record it until a witness says so.", "Get me someone who saw it. Nothing else will do."]},
    'farmer': {'low': ["Ask them down the lane. They will know.", "Someone closer will have it straight."],
               'medium': ["We should ask someone who was there.", "Better to ask than to guess."],
               'high': ["Ask someone who saw it, and quickly.", "Find a witness. I need to know what to plant."]},
    'smith': {'low': ["Ask the carters. They come through it.", "Someone on the road will know."],
              'medium': ["We should ask someone who was there.", "I would rather hear it from a witness."],
              'high': ["Get it from someone who saw. Then I will know what to make.", "Ask a witness, and do it today."]},
    'innkeeper': {'low': ["Someone in my room will have been there.", "I will ask tonight. Someone always knows."],
                  'medium': ["We should ask someone who was there.", "Let me put it to the room."],
                  'high': ["I will ask every traveller until one saw it.", "Find a witness. My tables will not settle otherwise."]},
    'miller': {'low': ["The carters will have it straighter.", "Ask whoever brought the grain."],
               'medium': ["We should ask someone who was there.", "Better to ask someone closer to it."],
               'high': ["Ask a witness before the sacks run out.", "Find someone who saw it. I must know what to hold back."]},
    'shepherd': {'low': ["The drovers will know. They cover the ground.", "Ask someone who walks that way."],
                 'medium': ["We should ask someone who was there.", "I would want it from a witness."],
                 'high': ["Ask someone who saw, before I move the flock.", "Find a witness. I will not risk them on a rumour."]},
    'woodcutter': {'low': ["Ask on the road. They carry it fresh.", "Someone coming down will know."],
                   'medium': ["We should ask someone who was there.", "Better to have it from a witness."],
                   'high': ["Ask someone who saw it before I go out again.", "Find a witness. I walk that road."]},
    'resident': {'low': ["We could ask someone who was there.", "Someone closer will know."],
                 'medium': ["We should ask someone who was there.", "Better to ask than to guess."],
                 'high': ["We should find a witness, and soon.", "Ask someone who saw it. I need to know."]},
}

# --- settle: agree to stop pursuing it for now ------------------------------
SETTLE = {
    'baker': {'low': ["Agreed. I have loaves waiting.", "Leave it there. The oven will not wait."],
              'medium': ["Agreed. Let us leave it there for now.", "That will do until we know more."],
              'high': ["Leave it. I cannot think about it and bake.", "Enough. I have work and no stomach for it."]},
    'scribe': {'low': ["Agreed. I will set it down as told.", "Let it rest there. I have it noted."],
               'medium': ["Agreed. Let us leave it there for now.", "I will leave the entry open."],
               'high': ["Leave it. I will write nothing I cannot stand behind.", "Enough for now. The page can wait."]},
    'farmer': {'low': ["Agreed. There is work in the field.", "Leave it there. The day is going."],
               'medium': ["Agreed. Let us leave it there for now.", "That will do until we hear more."],
               'high': ["Leave it be. Worrying will not fill a barn.", "Enough. I have the season to think of."]},
    'smith': {'low': ["Agreed. The fire is up.", "Leave it there. I have work."],
              'medium': ["Agreed. Let us leave it there for now.", "That will hold until we know better."],
              'high': ["Leave it. I would rather be doing than talking.", "Enough. I will make what is needed either way."]},
    'innkeeper': {'low': ["Agreed. The room wants serving.", "Leave it there. I will hear more by evening."],
                  'medium': ["Agreed. Let us leave it there for now.", "That will do. Someone will bring the rest."],
                  'high': ["Leave it. I have a full room and half a story.", "Enough. I will not have it repeated in here."]},
    'miller': {'low': ["Agreed. The stones are turning.", "Leave it there. I have grain waiting."],
               'medium': ["Agreed. Let us leave it there for now.", "That will do until the next load."],
               'high': ["Leave it. Talk grinds nothing.", "Enough. I have sacks to see to."]},
    'shepherd': {'low': ["Agreed. I should be back to them.", "Leave it there. The flock wants watching."],
                 'medium': ["Agreed. Let us leave it there for now.", "That will do until I hear more."],
                 'high': ["Leave it. I want them counted before dark.", "Enough. I will not stand here fretting."]},
    'woodcutter': {'low': ["Agreed. There is cutting to do.", "Leave it there. The light is going."],
                   'medium': ["Agreed. Let us leave it there for now.", "That will hold until I am back down."],
                   'high': ["Leave it. I would rather be inside before dusk.", "Enough. I have said all I know."]},
    'resident': {'low': ["Agreed. Let us leave it there.", "That will do for now."],
                 'medium': ["Agreed. Let us leave it there for now.", "That will do until we know more."],
                 'high': ["Leave it there. I have heard enough.", "Enough for now."]},
}

# --- part: sign off ---------------------------------------------------------
PART = {
    'baker': {'low': ["Right. The ovens want me.", "Good day to you."],
              'medium': ["I will keep an ear out.", "Mind how you go."],
              'high': ["I have said too much already. Good day.", "Go safely. Bar your door."]},
    'scribe': {'low': ["Noted. I will set it down as you told it.", "Very well."],
               'medium': ["I will keep the entry open.", "Good day to you."],
               'high': ["I will write what I can stand behind. No more.", "Go carefully."]},
    'farmer': {'low': ["Back to it, then.", "Good day."],
               'medium': ["I will hear more by market day.", "Mind how you go."],
               'high': ["God keep the season. Good day.", "Go safely."]},
    'smith': {'low': ["Aye. Back to the forge.", "Very well."],
              'medium': ["I will be here if you hear more.", "Mind yourself."],
              'high': ["I have heard enough. I will bar the door tonight.", "Go armed, if you are going far."]},
    'innkeeper': {'low': ["Come by tonight. I will have more.", "Good day to you."],
                  'medium': ["I will hear the rest by evening.", "Mind how you go."],
                  'high': ["I will keep the room quiet on it. Good day.", "Go safely, and do not travel late."]},
    'miller': {'low': ["Back to the stones.", "Good day."],
               'medium': ["Send word if you hear more.", "Mind how you go."],
               'high': ["I will hold back what I can. Good day.", "Go safely."]},
    'shepherd': {'low': ["I will be about the flock.", "Good day to you."],
                 'medium': ["I will keep watch up there.", "Mind how you go."],
                 'high': ["I am bringing them in early. Good day.", "Go safely, and keep off the high road."]},
    'woodcutter': {'low': ["Back up the hill for me.", "Very well."],
                   'medium': ["I will keep my eyes open on the road.", "Mind how you go."],
                   'high': ["I will not walk that road alone again. Good day.", "Go safely."]},
    'resident': {'low': ["Very well.", "Good day to you."],
                 'medium': ["I will keep an ear out.", "Mind how you go."],
                 'high': ["I have heard enough. Good day.", "Go safely."]},
}

# --- hedge: say how firmly the account is held ------------------------------
# Keyed by how the belief was acquired, then by stress. Epistemics, not trade,
# so voice does not key this one.
HEDGE = {
    'unsure': {'low': ["I am unsure of it. I would not swear to it.", "I could not say for certain."],
               'medium': ["I am unsure. It came to me thin.", "I would not stand behind it yet."],
               'high': ["I am unsure, and that is the worst of it.", "I do not know, and not knowing is half the fear."]},
    'retold': {'low': ["It has passed through several people by now.", "It is well travelled. Make of that what you will."],
               'medium': ["The account has passed through several people.", "It has been repeated often enough to drift."],
               'high': ["Everyone has it, and no two the same.", "It has been round the whole district. I trust none of it."]},
    'held': {'low': ["That is the account I hold.", "I have it as I told it."],
             'medium': ["That is the account I hold, and I have had no better.", "I stand by what I said."],
             'high': ["That is what I hold, and I would rather be wrong.", "I hold to it. I wish I did not."]},
}

# --- attribute: say where it came from --------------------------------------
ATTRIBUTE = {
    'unsure': ["It came to me second hand. I could not name who first.",
               "I could not tell you who carried it first."],
    'retold': ["It has come by so many mouths I could not name the first.",
               "Too many have carried it to say where it began."],
    'held': ["It came from someone who was there, as I understood it.",
             "I had it from someone close to it."],
}

# --- dispute: contradict with one's own account -----------------------------
DISPUTE_OPEN = {
    'low': ["I heard a different account. ", "That is not how it reached me. "],
    'medium': ["I heard it otherwise. ", "I have a different account. "],
    'high': ["That is not what I heard at all. ", "No. I had it quite differently. "],
}
DISPUTE_CLOSE = {
    'low': [" How sure are you?", " Which of us has it right?"],
    'medium': [" How sure are you?", " Where did yours come from?"],
    'high': [" How sure are you of yours?", " One of us has it wrong, and it matters."],
}

# --- situation: a predicament rewrites the opening --------------------------
# Hunger, exposure and travel mark a few moves the way stress marks a dispute.
# Only the marked state speaks up; every other state contributes the empty
# string, so the base pool passes through untouched and the target carries the
# predicament exactly when it holds. Three near-synonyms a cell, the
# STRESS_PREFIX discipline, trade-neutral like the stress pools.
SITUATION_MARKS = {
    ('hungry', True, 'hedge'): ["My belly has been empty two days. ",
                                "Hunger makes everything sound worse. ",
                                "I have not eaten since yesterday. "],
    ('hungry', True, 'settle'): ["I cannot think on an empty stomach. ",
                                 "We will settle nothing hungry. ",
                                 "My thoughts keep turning to food. "],
    ('sheltered', False, 'remark'): ["Another night with no roof. ",
                                     "The cold got in again last night. ",
                                     "I slept out, and it shows. "],
    ('sheltered', False, 'muse'): ["The nights out are getting to me. ",
                                   "I dream of a roof that holds. ",
                                   "Cold ground makes for cold thoughts. "],
    ('in_transit', True, 'open'): ["I only just came down the road. ",
                                   "I walked in with it this morning. ",
                                   "Fresh word, and I am already leaving. "],
    ('in_transit', True, 'defer'): ["I am only passing through. ",
                                    "Do not keep me. I have miles yet. ",
                                    "The road does not wait. "],
}


def situation_marks(move, situation):
    """Opening sentences the speaker's predicament contributes, or [''].

    Moves are disjoint across the table, so at most one cell matches; a move
    with no mark, or a row with no situation, composes over the empty string
    and comes out exactly as it went in.
    """
    for (axis, marked, marked_move), marks in SITUATION_MARKS.items():
        state = (situation or {}).get(axis)
        if marked_move == move and state is not None and bool(state) == marked:
            return marks
    return ['']


# --- company: who the speaker stands with and owes --------------------------
# Debts, trust and faction mark a few moves the way hunger marks a hedge. Same mechanic as SITUATION_MARKS, except faction is
# categorical (crown/guild/commons, None absent) so its cells key on the kind
# itself rather than a boolean. Only the marked state speaks; absence and the
# unmarked states contribute [''], and several marks may stack on one move
# (trust and faction both mark affirm), applied innermost first.
SOCIAL_MARKS = {
    ('owes_listener', True, 'defer'): ["I owe you a straight answer. ",
                                       "For you, I will ask it myself. ",
                                       "You are owed better than my guessing. "],
    ('owes_listener', True, 'settle'): ["We are square after this. ",
                                        "Call us even and leave it there. ",
                                        "Settled, and I owe you thanks for it. "],
    ('trusts_listener', True, 'affirm'): ["Between you and me, that is the account. ",
                                          "I would tell no one else. You have it right. ",
                                          "From you, I believe it. "],
    ('trusts_listener', True, 'attribute'): ["Between you and me. ",
                                             "You asked me straight. ",
                                             "For your ears only. "],
    ('faction', 'crown', 'affirm'): ["The crown hears the same. ",
                                     "It is known at court as you tell it. ",
                                     "The crown's word matches yours. "],
    ('faction', 'guild', 'affirm'): ["The guild books agree with you. ",
                                     "Our ledgers say the same. ",
                                     "The guild heard it likewise. "],
    ('faction', 'commons', 'affirm'): ["Every hearth says the same. ",
                                       "That is the talk at every table. ",
                                       "Common word agrees with you. "],
    ('faction', 'crown', 'part'): ["The crown thanks you. ",
                                   "Go with the crown's favor. ",
                                   "Court business calls me. "],
    ('faction', 'guild', 'part'): ["The guild owes you custom. ",
                                   "Trade calls me away. ",
                                   "Count it settled in the books. "],
    ('faction', 'commons', 'part'): ["Mind how you go. ",
                                     "Supper waits, and so does work. ",
                                     "Good day, and good neighbors. "],
    ('far_from_home', True, 'open'): ["Far from home, this is what I carry. ",
                                      "I bring word from further than here. ",
                                      "A traveller's news, take it as such. "],
    ('far_from_home', True, 'muse'): ["Home feels far tonight. ",
                                      "I wonder what they eat at home. ",
                                      "Distance makes everything urgent. "],
}


def social_marks(move, social):
    """All marked-state openings for this move, innermost first, or [''].

    Unlike situation marks, several cells can match one move (trust and
    faction both mark affirm); each wraps the previous, cross-product style,
    the way dispute crosses openers with closers.
    """
    out = ['']
    for (axis, marked, marked_move), marks in SOCIAL_MARKS.items():
        if marked_move != move:
            continue
        state = (social or {}).get(axis)
        if state is None or state != marked:
            continue
        out = [a + b for a in out for b in marks]
    return out


def tier(row):
    """How the belief was acquired: the cue the prompt already carries."""
    if row.get('confidence', 80) < 40: return 'unsure'
    if row.get('retold'): return 'retold'
    return 'held'


def forms(move, row, stance):
    """Every wording accepted for this cell.

    The generator samples from this and the grader accepts all of it. One
    function means a corpus and its scoring cannot drift apart -- which is how
    exact match against a single string quietly became a memorisation test.
    Returns None where the wording is the account itself rather than a pool.
    """
    voice = stance.get('voice') if stance.get('voice') in VOICES else 'resident'
    stress = stance.get('stress') if stance.get('stress') in STRESS else 'medium'
    table = {'affirm': AFFIRM, 'defer': DEFER, 'settle': SETTLE, 'part': PART}.get(move)
    if table is not None:
        return list(table[voice][stress])
    if move == 'hedge':
        return list(HEDGE[tier(row)][stress])
    if move == 'attribute':
        return list(ATTRIBUTE[tier(row)])
    return None


def choose(move, row, stance, rng):
    pool = forms(move, row, stance)
    return rng.choice(pool) if pool else None
