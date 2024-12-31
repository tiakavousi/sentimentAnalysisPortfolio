class TestCases :
    """
    contains a list of edge cases for sentiment analysis testing.
    Each case contains the text and expected sentiment.
    """
    edge_cases = [
        # Positive with Negation
        {
            'text': "I can't deny that this place is amazing. Not a single thing wrong with the food or service!",
            'expected': 'Positive'
        },
        {
            'text': "Never had a bad experience here. The food isn't anything less than perfect.",
            'expected': 'Positive'
        },
        {
            'text': "Not once have I been disappointed by their service. The staff isn't unfriendly at all.",
            'expected': 'Positive'
        },

        # Positive with Sarcasm
        {
            'text': "Yeah right, like this restaurant could possibly get ANY better! *chef's kiss* Absolutely perfect!",
            'expected': 'Positive'
        },
        {
            'text': "Oh sure, just RUIN my diet with your impossibly delicious desserts! How dare you be this good!",
            'expected': 'Positive'
        },
        {
            'text': "Just what I needed - another restaurant to be obsessed with! 🙄 Now I'll have to keep coming back!",
            'expected': 'Positive'
        },

        # Positive with Multipolarity
        {
            'text': "The wait was long but honestly worth every minute. Amazing food and exceptional service!",
            'expected': 'Positive'
        },
        {
            'text': "Small portions and pricey, but the taste makes up for everything. Will definitely return!",
            'expected': 'Positive'
        },
        {
            'text': "Noisy atmosphere but incredible food and the best service I've had in years.",
            'expected': 'Positive'
        },

        # Negative with Negation
        {
            'text': "The food isn't good at all. Not worth the price and I won't be returning.",
            'expected': 'Negative'
        },
        {
            'text': "I couldn't find anything special about this place. The service wasn't even close to acceptable.",
            'expected': 'Negative'
        },
        {
            'text': "Not once did they get our order right. The manager wasn't helpful either.",
            'expected': 'Negative'
        },

        # Negative with Sarcasm
        {
            'text': "Oh fantastic, another overpriced meal with cold food. Just what I was hoping for! 🙄",
            'expected': 'Negative'
        },
        {
            'text': "Wow, amazing how they consistently manage to mess up a simple order. Such talent! 😒",
            'expected': 'Negative'
        },
        {
            'text': "Five stars for teaching me the true meaning of patience! Best 2-hour wait ever! 🙃",
            'expected': 'Negative'
        },

        # Negative with Multipolarity
        {
            'text': "Great location but terrible food and even worse service. Definitely not returning.",
            'expected': 'Negative'
        },
        {
            'text': "Beautiful decor, shame about the rude staff and inedible food.",
            'expected': 'Negative'
        },
        {
            'text': "Nice ambiance but overpriced food and disappointing service ruined the experience.",
            'expected': 'Negative'
        },

        # Neutral with Negation
        {
            'text': "The food isn't amazing but isn't terrible either. Just an average experience.",
            'expected': 'Neutral'
        },
        {
            'text': "Not the best, not the worst. Wouldn't go out of my way to return.",
            'expected': 'Neutral'
        },
        {
            'text': "Can't say it was great, can't say it was bad. Just okay.",
            'expected': 'Neutral'
        },

        # Neutral with Sarcasm
        {
            'text': "Well, that was... an experience. I guess that's one way to run a restaurant! 🤔",
            'expected': 'Neutral'
        },
        {
            'text': "'Interesting' take on Italian food. Very... unique interpretation! 😏",
            'expected': 'Neutral'
        },
        {
            'text': "Such a... memorable experience. Definitely different from what I expected! 🫤",
            'expected': 'Neutral'
        },

        # Neutral with Multipolarity
        {
            'text': "Good food but slow service. Bad parking but nice location. Evens out I guess.",
            'expected': 'Neutral'
        },
        {
            'text': "Some dishes were great, others terrible. Service varied. Hard to form an opinion.",
            'expected': 'Neutral'
        },
        {
            'text': "Excellent appetizers, mediocre mains, poor desserts. Averages out to okay.",
            'expected': 'Neutral'
        }
    ]