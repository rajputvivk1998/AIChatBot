
import numpy as np

class Smileys:
    def __init__(self):
        self.smileys = {}

        ## Adding Emoticons
        self.add(":)", "midget smiley")
        self.add("^_^", "happy face")
        self.add(";)", "wink")
        self.add(":(", "sad")
        self.add(":'(", "crying")
        self.add(":-/", "sarcasm")
        self.add(":*", "kiss")
        self.add(":o", "surprise")
        self.add("(:-)", "smiley big face")
        self.add(",-}", "winking happy smiley")
        self.add("8-O", "Oh my good")
        self.add(":-*", "kiss")
        self.add(":->", "sarcastic smiley")
        self.add(":-V", "shouting smiley")
        self.add(":-X", "a big wet kiss")
        self.add(":-]", "smiley blockhead")
        self.add(";-(", "crying smiley")
        self.add(":-&", "tongue tied")
        self.add(":-<", "walrus smiley")
        self.add("+:)", "priest smiley")
        self.add("o:-)", "angel smiley")
        self.add(":-@", "screaming smiley")
        self.add("(-:", "left hand smiley")
        self.add(";^)", "smirking smiley")
        self.add(":-O", "talkative smiley")
       
 

    def add(self, icon, meaning):
        if icon not in self.smileys.keys():
            self.smileys[icon] = meaning

    def get_meaning(self, icon):
        return self.smileys[icon]

    def to_text(self, text):
        emojis = self.smileys.keys()
        if text in emojis:
            return self.smileys[text]
        else:
            return None
