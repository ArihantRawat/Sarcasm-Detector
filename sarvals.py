interjections = [ "aha", "ahem", "ahh", "ahoy", "alas", "arg", "aw", "bam", "bingo", "blah", "boo", "bravo", "brrr", 
                  "cheers", "congratulations", "dang", "drat", "darn", "duh","eek", "eh", "encore", "eureka", 
                  "fiddlesticks", "gadzooks", "gee", "gee" "whiz", "golly", "goodbye", "goodness", "good" "grief", "gosh",  
                  "ha-ha", "hallelujah", "hello", "hey", "hmm", "holy" "buckets", "holy" "cow", "holy" "smokes", "hot" "dog",
                  "huh", "humph", "hurray",  "oh", "oh" "dear", "oh" "my", "oh" "well", "oops", "ouch", "ow", "phew", "phooey", 
                  "pooh", "pow", "shh", "shoo", "thanks", "there", "tut-tut", "uh-huh", "uh-oh", "ugh", "wahoo", "well", "whoa", 
                  "whoops", "wow", "yeah", "yes", "yikes", "yippee", "yo", "yuck", "Whoop-de-doo" ]

emo_repl = {
    #good emotions
    "&lt;3" : " good ",
    ":d" : " good ",
    ":dd" : " good ",
    ":p" : " good ",
    "8)" : " good ",
    ":-)" : " good ",
    ":)" : " good ",
    ";)" : " good ",
    "(-:" : " good ",
    "(:" : " good ",
    
    "yay!" : " good ",
    "yay" : " good ",
    "yaay" : " good ",
    "yaaay" : " good ",
    "yaaaay" : " good ",
    "yaaaaay" : " good ",    
    #bad emotions
    ":/" : " bad ",
    ":&gt;" : " sad ",
    ":')" : " sad ",
    ":-(" : " bad ",
    ":(" : " bad ",
    ":s" : " bad ",
    ":-s" : " bad "
}

#general
re_repl = {
    r"\br\b" : "are",
    r"\bu\b" : "you",
    r"\bhaha\b" : "ha",
    r"\bhahaha\b" : "ha",
    r"\bdon't\b" : "do not",
    r"\bdoesn't\b" : "does not",
    r"\bdidn't\b" : "did not",
    r"\bhasn't\b" : "has not",
    r"\bhaven't\b" : "have not",
    r"\bhadn't\b" : "had not",
    r"\bwon't\b" : "will not",
    r"\bwouldn't\b" : "would not",
    r"\bcan't\b" : "can not",
    r"\bcannot\b" : "can not"    
}

emo_repl_order = [k for (k_len,k) in reversed(sorted([(len(k),k) for k in emo_repl.keys()]))]


intensifier_list = ["absolutely", "completely", "extremely", "highly", "rather", "really", "so", "too", "totally", "utterly",
                     "very", "awfully", "bloody", "dreadfully", "hella", "most", "precious", "quite", "reamarkably", "terribly",
                     "moderately", "fully", "super", "terrifically", "surpassingly", "excessively", "colossally", "frifhtfully",
                     "astoundingly", "phenomenally", "uncommonly", "outrageously", "fantastically", "mightily", "supremely", 
                     "insanely", "strikingly", "extraordinarily", "amazingly", "radically", "unusually", "exceptionally", 
                     "incredibly", "totally", "especially"]