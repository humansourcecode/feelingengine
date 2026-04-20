"""Mechanism Prompt Library — interview prompts + example responses.

Shared infrastructure for Beat Planning Sessions and any tool that guides
creators through mechanism-first content creation. Each mechanism has:

  - phenomenology: short description of the felt experience
  - interview_prompts: questions that elicit a story deploying this mechanism
  - sensory_followups: prompts to push for embodied/specific detail
  - depth_probes: prompts for surface-level responses that need more
  - example_responses: 3+ cross-domain sample stories demonstrating the
      mechanism in action (fictional but believable, spanning diverse
      life contexts)
  - mechanism_check_cue: question for verifying a response deploys the
      mechanism

These are interview scaffolds, not a script. A Beat Planner or facilitated
session uses them to elicit the creator's own story for each beat.

Exemplars are intentionally diverse: family, work, health, relationships,
sports, art, travel, childhood, loss, joy — so creators can pattern-match
from many life domains.
"""
from __future__ import annotations


PROMPTS: dict = {

    # ── I. Interoception mechanisms ─────────────────────────────────
    "body-turn": {
        "phenomenology": "attention drops into the body — breath catches, words recede",
        "interview_prompts": [
            "When did something drop — not figuratively, but literally in your body?",
            "When did the words stop mattering and you just felt it?",
            "What did your body know before your mind did?",
        ],
        "sensory_followups": [
            "Where in your body did you feel it?",
            "What were you doing with your hands at that moment?",
            "Was there a sound you registered before a thought?",
        ],
        "depth_probes": [
            "Tell me the version you haven't told anyone.",
            "What were you doing the minute before?",
        ],
        "example_responses": [
            "First time walking into my mom's empty house after she passed. I had a speech prepared, words I'd rehearsed. Opened the door and all the language left me. Just chest, weight, floor.",
            "Standing at the altar, watching her walk down the aisle. I'd practiced what to say. None of it was there. Just my heartbeat loud in my ears.",
            "Second week of chemo. The doctor said my numbers were moving the right way. I didn't understand what I was supposed to feel. But my shoulders — they fell two inches. My body knew before I did.",
            "Soccer final, penalty kick, I'm the goalkeeper. Ball comes off his foot. I dove — and I swear I felt it leave my fingers before I saw it go in the corner.",
        ],
        "mechanism_check_cue": "Does the response describe body-state taking precedence over language/thought?",
    },

    "body-surge": {
        "phenomenology": "something physical moves through — heart lifts or drops, shock lands",
        "interview_prompts": [
            "When did something physical move through you — a shock, a lift, a drop?",
            "Tell me about a moment you felt before you understood it.",
            "When did your body react before your mind caught up?",
        ],
        "sensory_followups": [
            "Where did you feel it first?",
            "How long before your thinking caught up?",
        ],
        "depth_probes": [
            "What did you do in that half-second?",
            "If you could slow-motion it, what came in what order?",
        ],
        "example_responses": [
            "The dog ran into the street. I was already moving before I understood what I'd seen. Cars stopped. I was standing in the middle of the road with her in my arms before I registered I was scared.",
            "Got the text — 'she said yes.' I felt it in my legs first. Had to sit down. Didn't even know I was grinning.",
            "Standing in line at the pharmacy. Saw the ex of ten years across the store. Heart did the thing. Then I realized it wasn't him — just somebody in the same hat. Body took three more minutes to come down.",
        ],
        "mechanism_check_cue": "Does the response show rapid involuntary physical reaction preceding conscious understanding?",
    },

    "body-anchor": {
        "phenomenology": "sustained body presence — breath slow, nothing else competes",
        "interview_prompts": [
            "When did time seem to slow because you were so present in your body?",
            "Tell me about a moment nothing competed with being in your body.",
            "When was the last time you held still inside your own skin?",
        ],
        "sensory_followups": [
            "What did your breath do?",
            "How long did this state last?",
        ],
        "depth_probes": [
            "What thoughts tried to come in that you let go of?",
            "What were you aware of that you don't usually notice?",
        ],
        "example_responses": [
            "Sitting with my newborn at 3am. Nobody was awake. She wasn't crying. I wasn't either. Just the sound of my own breathing and hers.",
            "After the surgery. Morphine drip. I wasn't thinking. Just noticing the rectangle of light on the ceiling.",
            "Hiking above the tree line, finally stopped at a switchback. Wind, no people. My lungs doing their thing.",
            "Swimming laps. Around lap 20, thoughts fall away. Just stroke, breath, the line at the bottom.",
        ],
        "mechanism_check_cue": "Does the response describe sustained embodied presence without narrative urgency?",
    },

    "sensation-flood": {
        "phenomenology": "the body is filled — sensory overwhelm, more than can be processed",
        "interview_prompts": [
            "When were you overwhelmed — not by emotion, by sensation?",
            "Tell me about a moment your senses couldn't keep up.",
            "When have you been more full than you could process?",
        ],
        "sensory_followups": [
            "Which sense was loudest?",
            "What did you stop being able to do?",
        ],
        "depth_probes": [
            "Did you try to narrate it to yourself, and fail?",
            "What's the part you still can't describe?",
        ],
        "example_responses": [
            "First time in Tokyo. Shibuya crossing at night. I wasn't scared, wasn't excited — just couldn't take it all in. Lights, people, language I couldn't read, signs in three directions, the smell of food and rain.",
            "A really good meal once, in Rome. I don't remember what it was. I remember not being able to talk.",
            "Kid's first concert — Springsteen. Arena. He was crying and I didn't know why. He said later, 'I just couldn't.'",
            "A good orgasm — the kind that take thirty seconds to come back from.",
        ],
        "mechanism_check_cue": "Does the response describe being flooded past the point of cognitive processing?",
    },

    # ── II. Core affect ────────────────────────────────────────────
    "affect-rise": {
        "phenomenology": "feeling thickens — the moment gets more charged",
        "interview_prompts": [
            "When did the air in a room change?",
            "Tell me about something getting heavier and heavier.",
            "When did a moment get more loaded than you expected?",
        ],
        "sensory_followups": [
            "When did you first notice the shift?",
            "What were other people doing?",
        ],
        "depth_probes": [
            "What was the first small sign?",
            "When could you have interrupted it and didn't?",
        ],
        "example_responses": [
            "Dinner with my parents. Small comment about my brother. Then another. Then my dad's voice shifted. You could feel the table tighten.",
            "Watching the game go to overtime. Up, down, up, down. By the time it was tied in the fourth quarter, my hands were clenched and I hadn't noticed.",
            "Board meeting going longer than it should. Two people disagreeing on something I thought was settled. Air gets thick.",
        ],
        "mechanism_check_cue": "Does the response show emotional intensity building gradually toward a charged state?",
    },

    "affect-fade": {
        "phenomenology": "the storm passes — settling, what's left is quieter",
        "interview_prompts": [
            "Tell me about the moment a storm passes.",
            "When did intensity give way to quiet?",
            "What's your sense of 'after'?",
        ],
        "sensory_followups": [
            "What did the quiet sound like?",
            "What was still on your body from the intensity?",
        ],
        "depth_probes": [
            "Was the quiet relief, or loss, or something else?",
            "What did you want to do, and what did you actually do?",
        ],
        "example_responses": [
            "Funeral was over. Everyone had left the house. Plates still on the counter. Just me, standing there, knowing I wasn't going to bed yet but also not doing anything.",
            "After the wedding reception ended. My shoes off, makeup still half on, sitting on the bed trying to remember what had just happened.",
            "Tour finished. Six months of flights. First morning home, I sat in my own kitchen and didn't know what to do with my hands.",
        ],
        "mechanism_check_cue": "Does the response describe emotional de-escalation after peak?",
    },

    # ── III. Regulation ────────────────────────────────────────────
    "restraint": {
        "phenomenology": "something withheld — a pressure that doesn't let out",
        "interview_prompts": [
            "When did you hold something back that wanted to come out?",
            "Tell me about a moment you kept composure and felt the cost.",
            "When have you been containing more than showed?",
        ],
        "sensory_followups": [
            "Where did the pressure sit in your body?",
            "What would have come out if you'd let it?",
        ],
        "depth_probes": [
            "What were you afraid would happen if you let go?",
            "Did anyone see it anyway?",
        ],
        "example_responses": [
            "Eulogy for my grandfather. I wrote it, practiced it, knew the lines. Got to the part about his hands. I had to pause, count to four, keep going. People thought I was moved. I was — but also doing math in my head to not collapse.",
            "Performance review. Manager said something about my 'potential' in a way that meant I was being passed over. I said 'thank you for the feedback.' I meant nothing I said that day.",
            "Holding my son while he got his vaccines. Four years old. Screaming. I was supposed to be smiling reassuringly. I smiled. My jaw hurt for an hour.",
        ],
        "mechanism_check_cue": "Does the response describe active suppression of an emotional response that was present?",
    },

    "release": {
        "phenomenology": "something let go — unguarded, an opening",
        "interview_prompts": [
            "When did something you'd been holding finally let go?",
            "Tell me about a moment of collapse — but the good kind.",
            "When did you stop holding on?",
        ],
        "sensory_followups": [
            "What did your body do in the first second after?",
            "How long had you been holding it?",
        ],
        "depth_probes": [
            "What was the trigger for letting go?",
            "What did the release feel different from what you expected?",
        ],
        "example_responses": [
            "Walked out of the job interview into the parking lot, got in the car, and started laughing. Then crying. Then laughing again. I didn't know which I was doing.",
            "Told the truth about an affair. Not proud. But after twelve months of holding it, the first thing I felt was like breathing out of a lung I hadn't used.",
            "She said 'you can put her down now.' I'd been holding the baby for an hour, frozen in case she woke up. Laid her in the crib. Arms finally let go. Realized they were shaking.",
        ],
        "mechanism_check_cue": "Does the response describe sudden regulatory collapse after sustained restraint?",
    },

    "threshold-approach": {
        "phenomenology": "something about to turn — tension winding, leaning in without arriving",
        "interview_prompts": [
            "When was the air about to break, but it hadn't yet?",
            "Tell me about a moment you could feel something coming.",
            "When did you know before it happened?",
        ],
        "sensory_followups": [
            "What were you looking at?",
            "What did you stop doing?",
        ],
        "depth_probes": [
            "How did you know it was coming?",
            "Did you want it to happen or not?",
        ],
        "example_responses": [
            "Three minutes before the verdict. Everyone in the courtroom still. Lawyer next to me had her pen uncapped and then capped. Judge's door opened.",
            "Second date. She said 'so...' and paused. I knew either a great question or a deal-breaker was next.",
            "Walking into the kitchen to tell my wife I'd lost the job. She was humming. Hadn't turned around yet.",
        ],
        "mechanism_check_cue": "Does the response capture tension without release — the moment before the shift?",
    },

    "withdrawal": {
        "phenomenology": "shrinking inward — body contracts in self-protection",
        "interview_prompts": [
            "When did you want to disappear?",
            "Tell me about the smallest you've felt in public.",
            "When did you pull back from something you'd already said?",
        ],
        "sensory_followups": [
            "Where did your body go small?",
            "What did your face do?",
        ],
        "depth_probes": [
            "What were you hoping no one would notice?",
            "When did you stop trying to recover?",
        ],
        "example_responses": [
            "Meeting, I proposed an idea I'd prepared for a week. Room was silent. Not hostile — just quiet. I spent the next forty minutes with my shoulders an inch lower.",
            "Mispronounced a word at dinner. His family noticed. His mom smiled kindly. I wanted the chair to swallow me.",
            "Sent the 'I love you' text first. Two hours went by. I kept opening my phone to check the send-receipt before the reply. Then I started typing a fake 'oh I meant to send that to my friend' but didn't send it.",
        ],
        "mechanism_check_cue": "Does the response describe self-protective inward contraction, distinct from active holding-back?",
    },

    # ── IV. Reward ─────────────────────────────────────────────────
    "anticipation": {
        "phenomenology": "leaning forward — wanting what's next, attention sharpens",
        "interview_prompts": [
            "When did you last hang on a next word?",
            "Tell me about a moment you wanted something to happen more than you wanted to eat.",
            "When were you bracing for something good?",
        ],
        "sensory_followups": [
            "What did you keep checking?",
            "What did time do?",
        ],
        "depth_probes": [
            "What were you imagining while waiting?",
            "When did the wanting shift from pleasure to tension?",
        ],
        "example_responses": [
            "Night before my daughter was born. Couldn't sleep. Kept imagining her face.",
            "Morning of the product launch. Checked my phone every ninety seconds.",
            "Boarding a flight to see someone I hadn't seen in eight years. Every row I passed was not their row, and the seat got closer.",
        ],
        "mechanism_check_cue": "Does the response show oriented, forward-leaning attention toward a coming reward?",
    },

    "satisfaction-peak": {
        "phenomenology": "a click — the answer lands, a small rest",
        "interview_prompts": [
            "When did a puzzle you'd been working on suddenly solve itself?",
            "Tell me about a moment everything made sense.",
            "What's the last time something clicked for you?",
        ],
        "sensory_followups": [
            "What did your body do in the first second after?",
            "Where did the tension go?",
        ],
        "depth_probes": [
            "How long had it been stuck?",
            "What was the tiny thing that tipped it?",
        ],
        "example_responses": [
            "Debugging for eleven hours. Went to bed. Woke up, typed one line, it worked.",
            "Trying to explain my therapist's insight to a friend. Mid-sentence I realized I'd been lying to myself for a decade.",
            "The last piece of the dissertation. I'd been hitting a wall for months. One afternoon I reread something from a 1978 paper and saw the connection.",
        ],
        "mechanism_check_cue": "Does the response describe a moment of resolution after sustained effort or buildup?",
    },

    # ── V. Memory ──────────────────────────────────────────────────
    "recognition": {
        "phenomenology": "wait — I know this — familiarity before full context",
        "interview_prompts": [
            "When did you recognize something before you'd remembered it?",
            "Tell me about a moment of 'I know this — I just don't know how.'",
            "When have you been sure without being able to say why?",
        ],
        "sensory_followups": [
            "What was the first sensory cue?",
            "How long before the memory surfaced?",
        ],
        "depth_probes": [
            "What other memories came with it?",
            "Was it good-familiar or unsettling-familiar?",
        ],
        "example_responses": [
            "Walking into a diner I'd never been to. Smelled like my grandmother's house. I couldn't place what specifically.",
            "Song came on in the grocery store. I knew the next lyric before I remembered hearing it in twenty years.",
            "Saw the number on the badge of a new coworker. My body reacted before I knew why. Turned out it was my childhood phone number.",
        ],
        "mechanism_check_cue": "Does the response describe felt-familiarity preceding conscious memory retrieval?",
    },

    "evocation": {
        "phenomenology": "a scene returns — not just memory, feeling again",
        "interview_prompts": [
            "What memory won't leave you alone?",
            "Tell me about a moment that came back uninvited.",
            "What do you think about without meaning to?",
        ],
        "sensory_followups": [
            "What triggers the return?",
            "Which sense carries the memory most strongly?",
        ],
        "depth_probes": [
            "Are you reliving it, or remembering it?",
            "What's the part that never changes?",
        ],
        "example_responses": [
            "The smell of chlorine. My father taught me to swim. He's been dead for fifteen years but the pool smell brings him back every time.",
            "My dog's collar is still on the hook. I see it, I'm thirty-four, I'm eight again, she's licking my face in the backyard.",
            "There's a phrase my ex used to say. I hear it in someone else's mouth twenty years later and I'm right back in that apartment.",
        ],
        "mechanism_check_cue": "Does the response describe memory arriving WITH its original affective charge, not just its content?",
    },

    "universal-recognition": {
        "phenomenology": "oh — that's true — body confirms without argument",
        "interview_prompts": [
            "What truth does your body confirm before your mind does?",
            "Tell me about something you said that made a room go still.",
            "What's something every human knows but rarely says out loud?",
        ],
        "sensory_followups": [
            "How did you know others felt it too?",
            "What made it unarguable?",
        ],
        "depth_probes": [
            "What conventional wisdom does this cut against?",
            "Why do you think people don't say it?",
        ],
        "example_responses": [
            "'No one wants to die.' Stops everyone. Bodies know.",
            "'Everyone is making it up as they go.' Every adult I've ever said this to has nodded.",
            "'Your parents didn't really know what they were doing.' Adults in their 30s: instant recognition. Their bodies relax because they've been carrying the assumption otherwise.",
            "'You will outlive everyone you love, or they will outlive you.' Nobody debates it. Everyone registers it.",
        ],
        "mechanism_check_cue": "Does the response invoke a claim the listener's body accepts as true before their mind evaluates it?",
    },

    # ── VI. Social ─────────────────────────────────────────────────
    "intimacy-turn": {
        "phenomenology": "something soft — room narrows, distance closes",
        "interview_prompts": [
            "Tell me about 'I'm going to tell you something I've never told anyone.'",
            "When did you decide in real time to let someone in?",
            "What was a moment of actually closing distance?",
        ],
        "sensory_followups": [
            "What did voice-volume do?",
            "What did eye contact do?",
        ],
        "depth_probes": [
            "What did you almost not say?",
            "Why did you decide, in that moment, to risk it?",
        ],
        "example_responses": [
            "Second bottle of wine with a friend I'd known since college. She said, 'I've never told anyone this, but...' and I knew the next hour was going to change our friendship.",
            "First time I told my wife what I was actually scared of. Not the thing I usually say. The real one.",
            "Dad on his last visit to see me in college. Car ride back from dinner. 'I wasn't a good father to you. I'm sorry.' I was twenty. I didn't know what to do with it.",
        ],
        "mechanism_check_cue": "Does the response describe a deliberate crossing toward another person's interior?",
    },

    "opposition": {
        "phenomenology": "positions collide — I am against you, the air tightens",
        "interview_prompts": [
            "When did you stop agreeing, and know you couldn't agree?",
            "Tell me about a moment you drew a line and held it.",
            "When have you been against someone, and stayed against?",
        ],
        "sensory_followups": [
            "What happened to your jaw / shoulders / stance?",
            "What did you stop being able to do (smile, nod)?",
        ],
        "depth_probes": [
            "What was at stake that you wouldn't concede?",
            "What was it like to not back down?",
        ],
        "example_responses": [
            "Family dinner, cousin said something about immigrants. I'd stayed quiet for years. That night I didn't. By dessert we were openly fighting.",
            "Boss told me to fire a person I thought was good. I said no. He said, 'you don't get to no me.' I said, 'watch me.'",
            "Partner said 'if we go to your mom's for Christmas, I'm not going.' We went back and forth for an hour. Nobody flinched.",
        ],
        "mechanism_check_cue": "Does the response show engaged, sustained disagreement — not avoidance, not capitulation?",
    },

    "vulnerability-transfer": {
        "phenomenology": "body opens because theirs is open — catching what they risk",
        "interview_prompts": [
            "When did someone else's vulnerability crack you open?",
            "Tell me about catching someone else's honesty.",
            "When did you feel what someone else was feeling in your own body?",
        ],
        "sensory_followups": [
            "When did you notice your own body shifting?",
            "What was the first sign they were risking something?",
        ],
        "depth_probes": [
            "Did you reciprocate? Why or why not?",
            "What did it cost them to say it?",
        ],
        "example_responses": [
            "Friend, crying on a video call. Said 'I think I'm not okay.' I didn't cry when she did. I cried six hours later in the shower.",
            "Public speech, speaker was a father whose son died. He said one line and his voice cracked. Auditorium of 400 people collectively did not breathe.",
            "First therapy session, therapist asked a gentle question, and something in how he asked it made me tell him something I'd never said out loud.",
        ],
        "mechanism_check_cue": "Does the response describe the listener's body responding to the speaker's risk?",
    },

    "boundary-establish": {
        "phenomenology": "a wall goes up — a line drawn, position stated",
        "interview_prompts": [
            "When did you say 'that's not who I am'?",
            "Tell me about a refusal you're proud of.",
            "When did you stop explaining and start declaring?",
        ],
        "sensory_followups": [
            "What was different about how you said it?",
            "What did your body do after?",
        ],
        "depth_probes": [
            "What had you been doing before the boundary?",
            "What did saying it cost or save you?",
        ],
        "example_responses": [
            "Thirty-fifth birthday. Decided: no more saying yes to things I don't want. First time I said no to a favor that week, felt weird. Then clean.",
            "Old friend asking me to lie on his resume. Two sentences: 'I can't do that. I'm sorry.'",
            "Kid asked if we could just skip church this week. I realized I'd been going for decades because I thought I had to. Said: 'Yeah. We can skip. We can always skip.'",
        ],
        "mechanism_check_cue": "Does the response describe self-definition through refusal, distinct from argumentative opposition?",
    },

    # ── VII. Language ──────────────────────────────────────────────
    "word-focus": {
        "phenomenology": "all attention on the words — thinking, not feeling",
        "interview_prompts": [
            "When did language matter more than feeling?",
            "Tell me about a time you were tracking words very closely.",
            "When have you been thinking-not-feeling?",
        ],
        "sensory_followups": [
            "What were you physically doing with your body while reading/listening?",
            "When did emotion try to intrude?",
        ],
        "depth_probes": [
            "What were you protecting yourself from feeling?",
            "When did you finally let yourself feel what you'd just understood?",
        ],
        "example_responses": [
            "Reading the contract, line by line. Couldn't have told you how I felt — just what the words said.",
            "Interpreting for my mother in a medical appointment. Doctor says X. I translate. I don't get to have my own reaction until we're in the car.",
            "Close-reading a poem for a grad seminar. I felt nothing. I was doing my job — parsing.",
        ],
        "mechanism_check_cue": "Does the response show deliberate language processing with affect held at bay?",
    },

    "word-recede": {
        "phenomenology": "the words stop mattering — what's said is less than what's felt",
        "interview_prompts": [
            "When did the words stop meaning anything to you?",
            "Tell me about a moment beyond speech.",
            "What's a time you forgot what someone said, but not how they said it?",
        ],
        "sensory_followups": [
            "What did you stop hearing?",
            "What were you aware of instead?",
        ],
        "depth_probes": [
            "Could you have said what you were feeling if asked?",
            "What was the moment words became inadequate?",
        ],
        "example_responses": [
            "Delivering my vows. Written them. Knew them. Looked at her. The rest of what I said, I don't remember.",
            "My grandfather, dying, trying to say something. I couldn't make out the words. It didn't matter. We were there.",
            "The silence after someone laughs really hard. They can't say what was funny anymore.",
        ],
        "mechanism_check_cue": "Does the response show language becoming secondary to felt experience?",
    },

    "contemplation": {
        "phenomenology": "sustained meaning-sift — thinking through, weighing",
        "interview_prompts": [
            "When have you thought hard about something for a long time?",
            "Tell me about turning an idea over in your hands.",
            "What's something you're still working through?",
        ],
        "sensory_followups": [
            "Where are you when you do this thinking?",
            "What body-state accompanies it?",
        ],
        "depth_probes": [
            "What do you know about it now that you didn't at the start?",
            "What's the part that still won't resolve?",
        ],
        "example_responses": [
            "Whether to have a second kid. I spent most of a year on walks working it out.",
            "Why I stopped calling my college friend. Took me five years to understand it wasn't him.",
            "Reading Epictetus on a park bench. One paragraph took me forty minutes.",
        ],
        "mechanism_check_cue": "Does the response describe sustained, non-anxious meaning-work on a complex question?",
    },

    # ── VIII. Multi-axis ──────────────────────────────────────────
    "inward-pivot": {
        "phenomenology": "external → internal — world falls away, it's about what's happening in me",
        "interview_prompts": [
            "When did the outside stop mattering?",
            "Tell me about a moment you turned inward — sharply.",
            "When did you leave the scene you were in?",
        ],
        "sensory_followups": [
            "What was happening externally that you stopped registering?",
            "When did you realize you'd left?",
        ],
        "depth_probes": [
            "What brought you back?",
            "Where did you go inside?",
        ],
        "example_responses": [
            "Noise all around at my dad's wake. I was shaking hands. At some point I wasn't there anymore. I was fourteen, him teaching me to tie a tie.",
            "Reading a sentence that suddenly described me. Stopped hearing the café. Stopped hearing the music. Just the paragraph.",
            "Meditation retreat. Third day. A bell rang and I didn't hear the bell. I heard the space around the bell.",
        ],
        "mechanism_check_cue": "Does the response describe attention moving from external events to internal experience?",
    },

    "pattern-break": {
        "phenomenology": "wait — what just happened — ground moves, attention redeploys",
        "interview_prompts": [
            "When did your assumptions collapse?",
            "Tell me about a moment the script flipped.",
            "What did you believe until you didn't?",
        ],
        "sensory_followups": [
            "What was the exact sentence or image that did it?",
            "How long did it take you to orient again?",
        ],
        "depth_probes": [
            "What had you believed, and why had you believed it?",
            "What was the first thing that made sense on the other side?",
        ],
        "example_responses": [
            "Teacher asked, 'what if you're wrong about the whole thing?' First time anyone had offered that as a possibility. I didn't sleep that night.",
            "Reading an email from my spouse's ex. Not bad. Not scandalous. Just... not what I'd been told the story was.",
            "CEO walked into the Monday meeting and said 'we're not doing this anymore.' Whole department's job was this.",
        ],
        "mechanism_check_cue": "Does the response describe a sudden reframe that required reorientation?",
    },

    "stakes-compression": {
        "phenomenology": "time is shorter than I thought — NOW matters, no more later",
        "interview_prompts": [
            "When did 'eventually' become 'now'?",
            "Tell me about time suddenly feeling shorter.",
            "What changed when you realized you were running out of time?",
        ],
        "sensory_followups": [
            "How did the pace of your decisions change?",
            "What did you drop doing?",
        ],
        "depth_probes": [
            "What had you been deferring?",
            "What became impossible to keep avoiding?",
        ],
        "example_responses": [
            "Doctor said five years, probably. I bought tickets to three countries within a week.",
            "Mother-in-law in the hospital. Weeks, not months. I cancelled everything. Every unwritten letter suddenly needed writing.",
            "Turned forty. Didn't feel different. But somehow the word 'someday' stopped meaning the same thing.",
        ],
        "mechanism_check_cue": "Does the response describe a felt collapse of time horizon, forcing urgency?",
    },

    "expansion": {
        "phenomenology": "outward opening — mind widens, something larger than me",
        "interview_prompts": [
            "What made you feel larger than yourself?",
            "Tell me about a moment your mind stretched.",
            "When did you feel the size of things?",
        ],
        "sensory_followups": [
            "What were you looking at?",
            "What happened to the body — did it feel smaller or part of something bigger?",
        ],
        "depth_probes": [
            "What word fell short?",
            "Did you want to tell someone, or keep it?",
        ],
        "example_responses": [
            "First time seeing the Milky Way away from city lights. I sat down.",
            "Pregnancy test positive. I didn't feel happy yet. I felt... bigger. Like I contained something.",
            "Watching the ocean at Cape Point. It's where two oceans meet. I cried, and I couldn't have told you why.",
        ],
        "mechanism_check_cue": "Does the response describe an experience of self being dwarfed or enlarged by something greater?",
    },

    "drift": {
        "phenomenology": "passive waiting — attention deployed minimally, time stretches",
        "interview_prompts": [
            "When did time stretch for no reason?",
            "Tell me about waiting — really waiting.",
            "What's a time nothing was happening but you were present?",
        ],
        "sensory_followups": [
            "What did you watch / listen to?",
            "How did you know time was passing?",
        ],
        "depth_probes": [
            "What were you half-thinking about?",
            "When was the last time you had nothing to do?",
        ],
        "example_responses": [
            "DMV line, an hour and a half. I watched the clock's second hand three times.",
            "Bus to the airport at 4am. Lights going past. Couldn't have told you what I was thinking.",
            "Chemo drip — forty-two minutes. I counted the ceiling tiles twice.",
        ],
        "mechanism_check_cue": "Does the response describe low-engagement waiting without narrative urgency?",
    },

    "dissonance": {
        "phenomenology": "something's off — cognitive mismatch, ironic register",
        "interview_prompts": [
            "When was something said that made everyone pretend they didn't notice?",
            "Tell me about a time the words and the feeling didn't match.",
            "When have you heard something where the truth was in the gap?",
        ],
        "sensory_followups": [
            "What did the faces in the room do?",
            "What would anyone have had to say for it to resolve?",
        ],
        "depth_probes": [
            "Did you call it out, or let it pass? Why?",
            "What would have happened if you'd named it?",
        ],
        "example_responses": [
            "Aunt, at grandma's funeral, said 'she lived a long life' with a smile that wasn't there for anything real.",
            "Boss, in the all-hands: 'We're all one family here.' Three layoffs announced that week.",
            "Ex said 'I'm really happy for you.' I knew she wasn't. She knew I knew. Neither of us said it.",
        ],
        "mechanism_check_cue": "Does the response describe mismatch between stated content and felt truth — the 'unspoken' register?",
    },
}


# Helper accessors for downstream code (Beat Planner, CLIs, etc.)

def get_prompts(mechanism: str) -> dict:
    """Return the full prompt record for a given mechanism, or {} if unknown."""
    return PROMPTS.get(mechanism, {})


def get_interview_prompts(mechanism: str) -> list:
    """Return just the interview_prompts list for a given mechanism."""
    return PROMPTS.get(mechanism, {}).get("interview_prompts", [])


def get_example_responses(mechanism: str) -> list:
    """Return example responses — for creator reference during interviews."""
    return PROMPTS.get(mechanism, {}).get("example_responses", [])


def all_mechanisms() -> list:
    """Return list of all mechanism names with prompts."""
    return sorted(PROMPTS.keys())
