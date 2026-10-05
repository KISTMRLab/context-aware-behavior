"""Write the authored starter dataset (Table 1 class labels) and its train/validation split.

Every sentence below was written and label-checked by hand for this repository; none
comes from the paper's unavailable 1,200-sentence dataset or from participants.
Labelling conventions:
  * Subject None = small talk, questions about the room, and negated requests
    ("Don't open the window"): no action is performed. Action/Position/Target are None.
  * Position is the spatial relation attached to the target or destination: on/onto -> On,
    in/into/inside -> In, to/towards/over to (incl. "to me") -> To, left/right side or hand
    -> Left/Right. Verb particles ("turn on") are part of the Action, not a Position.
  * Put labels the destination as Target ("put the pillow on the bed" -> Put/On/Bed);
    "put the pillow down" labels the carried object (Put/None/Pillow).
  * The generic Object class covers the room's small carryable items (book, box, cup...).
Run: python scripts/build_starter_dataset.py   (rewrites src/context_behavior/resources/starter/)
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from context_behavior.data import make_record  # noqa: E402
from context_behavior.ontology import Ontology, check_table1  # noqa: E402

OUT = ROOT / "src" / "context_behavior" / "resources" / "starter"
N = "None"

ACTIONS: dict[tuple[str, str, str], list[str]] = {
    # ---- Walk ------------------------------------------------------------------------------
    ("Walk", N, N): ["Walk around a little", "Take a few steps", "Could you walk around the room?",
                     "Go for a little stroll", "Move a bit", "Come here", "Come over here please",
                     "Step forward"],
    ("Walk", "To", "Bed"): ["Go to the bed", "Walk over to the bed", "Please head to the bed", "Walk towards the bed"],
    ("Walk", N, "Bed"): ["Approach the bed", "Go near the bed"],
    ("Walk", "To", "Window"): ["Walk to the window", "Go over to the window please", "Could you move to the window?",
                               "Head towards the window"],
    ("Walk", N, "Window"): ["Approach the window"],
    ("Walk", "To", "Chair"): ["Go to the chair", "Walk to the chair", "Please step over to the chair"],
    ("Walk", "To", "Drawer"): ["Walk to the drawer", "Go to the dresser", "Head over to the drawers"],
    ("Walk", "To", "Lamp"): ["Go to the lamp", "Walk towards the lamp"],
    ("Walk", "To", "Curtain"): ["Walk to the curtains", "Go over to the curtain"],
    ("Walk", "To", "Switch"): ["Go to the light switch", "Walk to the switch"],
    ("Walk", "Left", N): ["Walk to the left", "Move left", "Take a few steps to your left", "Step to the left please",
                          "Go left a bit"],
    ("Walk", "Right", N): ["Walk to the right", "Move right", "Step to the right", "Go a little to the right",
                           "Could you move over to the right?"],
    ("Walk", "Left", "Bed"): ["Go to the left side of the bed", "Walk around to the left of the bed"],
    ("Walk", "Right", "Bed"): ["Go to the right side of the bed", "Walk around to the right of the bed",
                               "Stand on the right side of the bed"],
    ("Walk", "Right", "Chair"): ["Stand on the right side of the chair", "Go to the right of the chair"],
    # ---- Run -------------------------------------------------------------------------------
    ("Run", N, N): ["Run", "Run around the room", "Jog in place for a bit", "Can you run?", "Start running",
                    "Sprint across the room"],
    ("Run", "To", "Window"): ["Run to the window", "Hurry over to the window", "Rush to the window please"],
    ("Run", "To", "Bed"): ["Run to the bed", "Jog over to the bed", "Dash to the bed"],
    ("Run", "To", "Chair"): ["Run to the chair", "Hurry to the chair"],
    ("Run", "To", "Drawer"): ["Run over to the drawer"],
    ("Run", "Left", N): ["Run to the left", "Jog left"],
    ("Run", "Right", N): ["Run to the right", "Sprint to the right side"],
    # ---- Open ------------------------------------------------------------------------------
    ("Open", N, "Drawer"): ["Open the drawer", "Please open the drawer", "Could you open the drawer for me?",
                            "Pull out the drawer", "Can you pull the drawer open?", "I need something from the drawer, open it",
                            "Would you mind opening the dresser?", "Open up the top drawer", "Slide the drawer open"],
    ("Open", N, "Window"): ["Open the window", "Please open the window", "It's stuffy in here, open the window",
                            "Could you open the window a bit?", "Let some fresh air in by opening the window",
                            "I'd like you to open the window", "Open the windows please", "Can you open that window?",
                            "Hey, open the window"],
    ("Open", N, "Curtain"): ["Open the curtains", "Please open the curtain", "Draw back the curtains",
                             "It's too dark, open the curtains", "Could you pull the curtains open?",
                             "Open the blinds", "Let the sunlight in, open the drapes"],
    ("Open", "Left", "Window"): ["Open the window with your left hand"],
    ("Open", "Right", "Drawer"): ["Use your right hand to open the drawer", "Open the drawer with your right hand"],
    # ---- Close -----------------------------------------------------------------------------
    ("Close", N, "Drawer"): ["Close the drawer", "Please shut the drawer", "Push the drawer in",
                             "Could you close the drawer?", "The drawer is open, close it please",
                             "Shut the dresser drawer", "Can you push in the drawer?"],
    ("Close", N, "Window"): ["Close the window", "Shut the window please", "I'm cold, close the window",
                             "Could you shut that window?", "It's noisy outside, close the window",
                             "Please close the windows", "Can you close the window for me?", "Close the window now"],
    ("Close", N, "Curtain"): ["Close the curtains", "Please close the curtain", "It's too bright, close the curtains",
                              "Shut the blinds", "Could you pull the curtains shut?", "Close the drapes please",
                              "I want some privacy, close the curtains"],
    ("Close", "Left", "Curtain"): ["Close the curtain with your left hand"],
    ("Close", "Right", "Window"): ["Shut the window with your right hand"],
    # ---- Sit -------------------------------------------------------------------------------
    ("Sit", N, N): ["Sit down", "Please sit down", "Have a seat", "Take a seat", "Why don't you sit down?",
                    "You can sit now", "Be seated please"],
    ("Sit", "On", "Chair"): ["Sit on the chair", "Please sit on the chair", "Have a seat on the chair",
                             "Could you sit down on the chair?", "Go sit on the chair", "Take a seat on that chair",
                             "You look tired, sit on the chair", "Sit on the stool", "Can you sit on the chair for me?",
                             "Sit down on the seat"],
    ("Sit", "In", "Chair"): ["Sit in the chair", "Take a seat in the armchair", "Please sit in that chair"],
    ("Sit", "On", "Bed"): ["Sit on the bed", "Have a seat on the bed", "Sit down on the edge of the bed",
                           "Could you sit on the bed?", "Go and sit on the bed"],
    ("Sit", "On", "Floor"): ["Sit on the floor", "Sit down on the ground", "Please sit on the floor",
                             "Take a seat on the floor", "Could you sit on the ground?"],
    # Requests the room cannot satisfy are still labelled as asked; the planner rejects them by affordance.
    ("Sit", "In", "Drawer"): ["Sit inside the drawer", "Go sit in the dresser", "Could you sit in the drawer?",
                              "Take a seat inside the drawers"],
    ("Sit", "On", "Drawer"): ["Sit on the drawer", "Have a seat on top of the dresser"],
    ("Sit", "On", "Lamp"): ["Sit on the lamp"],
    ("Sit", "In", "Window"): ["Sit in the window"],
    # ---- Stand up --------------------------------------------------------------------------
    ("Stand up", N, N): ["Stand up", "Please stand up", "Get up", "Could you stand up?", "Stand", "Rise",
                         "Get up please", "Time to get up", "On your feet please", "Up you get"],
    ("Stand up", N, "Chair"): ["Stand up from the chair", "Get up from the chair", "Get off the chair"],
    ("Stand up", N, "Bed"): ["Get out of bed", "Get off the bed", "Get up from the bed"],
    ("Stand up", N, "Floor"): ["Get up off the floor"],
    # ---- Turn on ---------------------------------------------------------------------------
    ("Turn on", N, "Lamp"): ["Turn on the lamp", "Please turn the lamp on", "Switch on the lamp",
                             "It's dark in here, turn on the light", "Could you switch the light on?",
                             "Turn the light on", "Power on the lamp", "Can you turn on the desk lamp?",
                             "I can't see, turn on the lamp", "Light up the lamp please", "Turn on the reading light"],
    ("Turn on", N, "Switch"): ["Flip the light switch on", "Turn on the switch", "Press the switch to turn it on",
                               "Switch on the wall switch", "Turn the light switch on", "Could you flip on the switch?"],
    ("Turn on", "Left", "Lamp"): ["Turn on the lamp with your left hand"],
    ("Turn on", "Right", "Switch"): ["Use your right hand to turn on the switch"],
    # ---- Turn off --------------------------------------------------------------------------
    ("Turn off", N, "Lamp"): ["Turn off the lamp", "Please turn the lamp off", "Switch off the lamp",
                              "Turn off the light", "Could you switch the light off?", "It's too bright, turn off the lamp",
                              "Shut off the lamp", "Power off the lamp", "I want to sleep, turn the light off",
                              "Turn the lamp off, it is on", "Can you turn off the bedside lamp?"],
    ("Turn off", N, "Switch"): ["Flip the light switch off", "Turn off the switch", "Switch off the wall switch",
                                "Turn the light switch off", "Could you flip off the switch?", "Press the switch to turn it off"],
    ("Turn off", "Left", "Lamp"): ["Turn the lamp off with your left hand"],
    ("Turn off", "Right", "Lamp"): ["Switch off the lamp using your right hand"],
    # ---- Lay -------------------------------------------------------------------------------
    ("Lay", N, N): ["Lie down", "Please lie down", "Lay down", "Take a nap", "Go to sleep", "Stretch out and rest"],
    ("Lay", "On", "Bed"): ["Lie on the bed", "Lay on the bed", "Please lie down on the bed", "Go lie on the bed",
                           "You look sleepy, lie on the bed", "Could you lay down on the bed?", "Take a nap on the bed",
                           "Stretch out on the bed", "Lie down on the mattress", "Go to bed and lie down on it"],
    ("Lay", "On", "Floor"): ["Lie on the floor", "Lay down on the floor", "Please lie on the ground",
                             "Could you lie down on the floor?"],
    ("Lay", "In", "Drawer"): ["Lie in the drawer", "Lie down inside the dresser"],
    ("Lay", "On", "Chair"): ["Lie on the chair", "Lay down on the chair"],
    # ---- Idle ------------------------------------------------------------------------------
    ("Idle", N, N): ["Relax", "Just relax", "Wait there", "Stay where you are", "Stay put", "Do nothing for a while",
                     "Chill for a moment", "Hold still", "Wait a moment please", "Just stay there and rest",
                     "Idle for now", "Take it easy"],
    # ---- Bring -----------------------------------------------------------------------------
    ("Bring", N, "Pillow"): ["Bring me the pillow", "Bring the pillow", "Could you fetch the pillow?",
                             "Hand me the pillow please", "Get me a pillow", "Can you bring me that pillow?",
                             "I need a cushion, bring it here", "Pass me the pillow", "Give me the pillow please"],
    ("Bring", "To", "Pillow"): ["Bring the pillow to me", "Carry the pillow over to me", "Bring the cushion over here to me"],
    ("Bring", N, "Object"): ["Bring me the book", "Fetch the box", "Could you bring me the object?",
                             "Hand me the book please", "Get me the cup", "Pass me that thing", "Give me the item on the floor",
                             "Can you fetch the book for me?"],
    ("Bring", "To", "Object"): ["Bring the book to me", "Carry the box over to me"],
    ("Bring", "Left", "Pillow"): ["Bring me the pillow in your left hand"],
    # ---- Hold ------------------------------------------------------------------------------
    ("Hold", N, "Pillow"): ["Hold the pillow", "Pick up the pillow", "Grab the pillow", "Could you hold the cushion?",
                            "Take the pillow", "Lift the pillow up"],
    ("Hold", N, "Object"): ["Hold the book", "Pick up the box", "Grab the book please", "Pick up that object",
                            "Take the cup", "Carry the book", "Can you pick up the item?"],
    ("Hold", "Left", "Object"): ["Hold the book with your left hand", "Pick up the box in your left hand"],
    ("Hold", "Right", "Pillow"): ["Hold the pillow in your right hand", "Grab the pillow with your right hand"],
    ("Hold", "Left", "Pillow"): ["Hold the pillow with your left hand"],
    # ---- Put -------------------------------------------------------------------------------
    ("Put", "On", "Bed"): ["Put the pillow on the bed", "Place the pillow on the bed", "Put it on the bed",
                           "Could you put the book on the bed?", "Set the cushion on the bed", "Leave the pillow on the bed"],
    ("Put", "On", "Chair"): ["Put the book on the chair", "Place the pillow on the chair", "Put it on the chair please",
                             "Set the box on the chair"],
    ("Put", "On", "Floor"): ["Put the book on the floor", "Place the pillow on the floor", "Put it on the ground",
                             "Set the box down on the floor"],
    ("Put", "In", "Drawer"): ["Put the book in the drawer", "Place the box inside the drawer", "Put it into the drawer",
                              "Could you put the cup in the drawer?", "Store the book in the dresser"],
    ("Put", "On", "Drawer"): ["Put the book on the drawer", "Place the box on top of the dresser", "Set the cup on the drawer"],
    ("Put", "Left", "Bed"): ["Put the book to the left of the bed", "Place the box on the left side of the bed"],
    ("Put", "Right", "Chair"): ["Put the pillow to the right of the chair", "Place the book on the right side of the chair"],
    ("Put", "Left", "Drawer"): ["Put the box to the left of the drawer"],
    ("Put", N, "Pillow"): ["Put the pillow down", "Drop the pillow", "Set the pillow down please"],
    ("Put", N, "Object"): ["Put the book down", "Drop the box", "Set the cup down"],
    ("Put", N, N): ["Put it down", "Drop it", "Put that down please"],
}

CONVERSATION = [
    # small talk
    "Hello", "Hi there", "Hey, how are you?", "Good morning", "Good evening", "How are you today?",
    "What's your name?", "Nice to meet you", "Who are you?", "How old are you?", "Where are you from?",
    "What do you like to do?", "Tell me a joke", "Thank you", "Thanks a lot", "You're welcome", "Goodbye",
    "See you later", "Have a nice day", "I'm fine, thanks", "I had a long day", "I'm a bit tired today",
    "What time is it?", "What's the weather like?", "Do you like music?", "What is your favorite color?",
    "Can you hear me?", "Are you a robot?", "That's interesting", "I see", "Okay", "Cool",
    "Tell me about yourself", "What are you doing?", "How was your day?", "Do you have any hobbies?",
    "I like this room", "You look nice today", "Are you happy?", "What makes you laugh?", "Sorry about that",
    "Never mind", "Let's talk for a while", "Do you know any stories?", "I'm bored", "What should we talk about?",
    "Can you speak Korean?", "Do you sleep at night?", "Where do you live?", "Is it going to rain tomorrow?",
    "What did you eat today?", "I love sunny days", "Do you have friends?", "How do you feel?", "That's funny",
    # questions and remarks about the room (no request)
    "Is the window open?", "Is the lamp on?", "Is the light switch on?", "What color is the chair?",
    "Where is the pillow?", "Is the drawer open?", "Are the curtains closed?", "What is in the drawer?",
    "How big is the bed?", "Is that your bed?", "Do you like the lamp?", "The curtains look nice",
    "Is the chair comfortable?", "Who bought this lamp?", "That pillow looks soft", "Why is the window so small?",
    "How many drawers are there?", "Whose book is that?", "I like your chair", "Is the bed comfortable?",
    "What is that object on the floor?", "The room is quite bright", "The window has a nice view",
    "Do you sleep on that bed?", "When did you open the window?",
    # negated or declined requests: no action
    "Don't open the window", "Do not close the curtains", "Please don't sit on the bed", "Don't turn on the lamp",
    "Never mind, don't bring the pillow", "You don't need to open the drawer", "Don't turn off the light",
    "Do not run in the room", "Don't lie on the floor", "No need to close the window", "Don't stand up yet",
    "Don't touch the switch", "Please do not pick up the book", "You shouldn't sit on the floor",
    "Don't put the pillow on the chair", "No, don't open the curtains", "Don't walk to the window",
    "There is no need to turn the lamp on", "Don't move the chair", "Please don't close the drawer",
]


def build():
    ontology = Ontology.load()
    check_table1(ontology)
    train, val, seen = [], [], set()
    groups = [((N, N, N, N), CONVERSATION)] + [(("Virtual Human",) + key, texts) for key, texts in ACTIONS.items()]
    for (subject, action, position, target), texts in groups:
        for index, text in enumerate(texts):
            if text.casefold() in seen:
                raise ValueError(f"duplicate sentence: {text}")
            seen.add(text.casefold())
            row = {"text": text, "subject": subject, "action": action, "position": position, "target": target}
            make_record(row, ontology)  # validates labels against the ontology and data rules
            (val if len(texts) >= 4 and index % 5 == 2 else train).append(row)
    return ontology, train, val


def main():
    ontology, train, val = build()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", train), ("val.jsonl", val)):
        (OUT / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    counts = {head: dict(sorted(Counter(r[head] for r in train + val).items())) for head in ("subject", "action", "position", "target")}
    missing = {head: sorted(set(ontology.classes[head]) - set(counts[head])) for head in counts}
    provenance = {
        "name": "context-aware-behavior authored starter dataset",
        "origin": "Written and label-checked by hand for this repository (scripts/build_starter_dataset.py). "
                  "It is not the paper's 1,200-sentence room dataset, which is unavailable, and contains no participant data.",
        "license": "CC0 1.0 Universal (public-domain dedication) for the authored sentences and labels",
        "labels": "Table 1 classes: Subject, Action, Position, Target (src/context_behavior/resources/ontology.json)",
        "split": "Per label combination, every fifth sentence (index 2, 7, ...) of groups with at least four sentences goes to validation",
        "train_examples": len(train), "val_examples": len(val), "class_counts": counts, "classes_without_examples": missing,
        "intended_use": "Starter data for the demo and for checking the training pipeline; extend it with sentences "
                        "reviewed for your own scene. Validation accuracy on this split is informational only.",
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: provenance[k] for k in ("train_examples", "val_examples", "classes_without_examples")}, indent=2))


if __name__ == "__main__":
    main()
