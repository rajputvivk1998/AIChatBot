import wikipedia

def get_summary(title):
    return wikipedia.summary(title, sentences=2)
